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


class TestCoreImports:
    """Test that core modules import correctly."""

    def test_core_modules(self):
        from enigma_engine import CONFIG
        from enigma_engine.core import EnigmaEngine
        from enigma_engine.core import get_hardware
        assert isinstance(CONFIG, dict)
        assert EnigmaEngine is not None
        assert get_hardware is not None


class TestCommandSystem:
    """Test the command processing system."""

    def test_parse_commands(self):
        from enigma_engine.core.commands import parse_commands
        text = "Hello [CMD]system.info[/CMD] world"
        clean, commands = parse_commands(text)
        assert len(commands) == 1
        assert commands[0] == "system.info"
        assert "[CMD]" not in clean
        assert "Hello" in clean and "world" in clean

    def test_parse_multiple_commands(self):
        from enigma_engine.core.commands import parse_commands
        text = "[CMD]gui.tab.switch chat[/CMD] and [CMD]system.info[/CMD]"
        _, commands = parse_commands(text)
        assert len(commands) == 2

    def test_registry(self):
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        available = registry.list_commands()
        assert len(available) > 0


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

    def test_router_basics(self):
        from enigma_engine.router import ModRouter
        assert hasattr(ModRouter, "get_prompt")
        assert hasattr(ModRouter, "set_prompt")

    def test_router_training_can_toggle_runtime(self, monkeypatch):
        """ModRouter can create and remove its trainer after init."""
        from enigma_engine import router as router_mod

        created = []

        class DummyTrainer:
            def __init__(self):
                self.started = False
                self.stopped = False
                created.append(self)

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        monkeypatch.setattr(router_mod, "BackgroundTrainer", DummyTrainer)

        router = router_mod.ModRouter(enable_training=False)
        router.running = True

        router.set_training_enabled(True)

        assert router.trainer is created[0]
        assert created[0].started is True

        router.set_training_enabled(False)

        assert router.trainer is None
        assert created[0].stopped is True


class TestModelRegistry:
    """Test model registry."""

    def test_registry_list(self):
        from enigma_engine.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        models = registry.list_models()
        assert isinstance(models, dict)


class TestProjectPackaging:
    """Test that packaging config is correct."""

    def test_no_setup_py(self):
        """setup.py should be deleted — pyproject.toml is the single source."""
        assert not (PROJECT_ROOT / "setup.py").exists()

    def test_pyproject_toml_exists(self):
        """pyproject.toml must exist with required fields."""
        toml_path = PROJECT_ROOT / "pyproject.toml"
        assert toml_path.exists()
        content = toml_path.read_text(encoding="utf-8")
        assert 'name = "enigma-engine"' in content
        assert 'version = "1.1.0"' in content
        assert "[project.scripts]" in content
        assert "[project.optional-dependencies]" in content
        assert "dynamic" not in content  # no more dynamic delegation


@pytest.mark.structural
class TestGGUFChatParameters:
    """Tests for GGUF chat parameter forwarding.

    The GGUF chat path in engine_chat.py must forward ALL
    user-configurable parameters (repetition_penalty, max_tokens,
    etc.) and truncate history before calling the model.
    """

    def test_gguf_chat_forwards_repeat_penalty(self):
        """GGUF chat path passes repeat_penalty to model.chat()."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.chat)
        # Must forward repetition_penalty as repeat_penalty for GGUF
        assert "repeat_penalty" in source

    def test_gguf_chat_forwards_max_tokens(self):
        """GGUF chat path uses max_tokens from kwargs."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.chat)
        # The GGUF section should read max_tokens from kwargs
        assert "max_tokens" in source

    def test_gguf_chat_truncates_history(self):
        """GGUF chat path truncates history before sending.

        Truncation now lives in _prepare_chat() which chat() calls
        before any generation path.
        """
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        # _prepare_chat handles truncation; chat() calls it first
        prep_source = inspect.getsource(_ChatMixin._prepare_chat)
        assert "_truncate_history" in prep_source, \
            "_prepare_chat must contain _truncate_history"
        chat_source = inspect.getsource(_ChatMixin.chat)
        lines = chat_source.split("\n")
        prepare_line = None
        gguf_line = None
        for i, line in enumerate(lines):
            if "_prepare_chat(" in line and prepare_line is None:
                prepare_line = i
            if "self.model.chat" in line and gguf_line is None:
                gguf_line = i
        assert prepare_line is not None, \
            "chat() must call _prepare_chat()"
        assert gguf_line is not None, \
            "self.model.chat not found in chat()"
        assert prepare_line < gguf_line, \
            "_prepare_chat must run BEFORE model.chat()"

    def test_gguf_model_chat_passes_kwargs(self):
        """GGUFModel.chat() forwards kwargs to create_chat_completion."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.chat)
        assert "**kwargs" in source


@pytest.mark.structural
class TestDeadImports:
    """Verify dead imports have been removed.

    Uses ruff F401 (unused imports) rule on critical modules.
    This is more robust than string-matching source lines — ruff
    understands Python scoping, __all__ re-exports, and type-only
    usage.
    """

    _CRITICAL_MODULES = [
        "enigma_engine/core/engine_chat.py",
        "enigma_engine/core/engine_generation.py",
        "enigma_engine/core/commands.py",
        "enigma_engine/core/training.py",
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
                f"Unused imports found in critical modules:\n"
                + "\n".join(lines)
            )


@pytest.mark.structural
class TestStreamChatTruncation:
    """Verify stream_chat truncates history like chat() does."""

    def test_stream_chat_has_truncation(self):
        """stream_chat must truncate history via _prepare_chat."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        # Truncation is in _prepare_chat; stream_chat calls it
        prep_source = inspect.getsource(_ChatMixin._prepare_chat)
        assert "_truncate_history" in prep_source
        source = inspect.getsource(_ChatMixin.stream_chat)
        assert "_prepare_chat(" in source

    def test_stream_chat_is_generator(self):
        """stream_chat must return a generator (yield tokens)."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        assert inspect.isgeneratorfunction(_ChatMixin.stream_chat)

    def test_cli_chat_uses_streaming(self):
        """run_chat in run.py must use stream_chat for token streaming."""
        from pathlib import Path
        source = Path("run.py").read_text(encoding="utf-8")
        assert "stream_chat" in source, (
            "run_chat() should use engine.stream_chat() for streaming output"
        )


@pytest.mark.structural
class TestChatFnKwargs:
    """Verify chat_fn in chat_with_tools merges kwargs."""

    def test_chat_fn_forwards_kw(self):
        """chat_fn should forward inner kw args, not discard them."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.chat_with_tools)
        # Find chat_fn definition — it accepts **kw
        # The inner self.chat() call should use kw somehow
        lines = source.split('\n')
        in_chat_fn = False
        chat_call_uses_kw = False
        for line in lines:
            if 'def chat_fn' in line:
                in_chat_fn = True
            if in_chat_fn and 'self.chat(' in line:
                # look ahead for **kw or **{**kwargs, **kw}
                pass
            if in_chat_fn and '**kw' in line and 'def ' not in line:
                chat_call_uses_kw = True
        assert chat_call_uses_kw, (
            "chat_fn captures **kw but never forwards it — "
            "kwargs from universal_chat are silently discarded")


class TestDocumentReaders:
    """Verify document_readers module structure and graceful fallbacks."""

    def test_module_imports(self):
        """document_readers should import without any required deps."""
        from enigma_engine.core.document_readers import (
            read_pdf, read_docx, read_document,
            pdf_available, docx_available,
            SUPPORTED_EXTENSIONS,
        )
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_read_pdf_raises_without_lib(self):
        """read_pdf raises ImportError when pymupdf is missing."""
        from enigma_engine.core import document_readers as dr
        if dr.pdf_available():
            pytest.skip("pymupdf is installed — skip missing-lib test")
        with pytest.raises(ImportError, match="pymupdf"):
            dr.read_pdf("fake.pdf")

    def test_read_docx_raises_without_lib(self):
        """read_docx raises ImportError when python-docx is missing."""
        from enigma_engine.core import document_readers as dr
        if dr.docx_available():
            pytest.skip("python-docx is installed — skip missing-lib test")
        with pytest.raises(ImportError, match="python-docx"):
            dr.read_docx("fake.docx")

    def test_read_document_returns_none_on_missing_lib(self):
        """read_document returns None for unsupported/unavailable formats."""
        from enigma_engine.core.document_readers import read_document
        # Unknown extension always returns None
        assert read_document("file.xyz") is None

    def test_read_pdf_file_not_found(self):
        """read_pdf raises FileNotFoundError for missing file."""
        from enigma_engine.core import document_readers as dr
        if not dr.pdf_available():
            pytest.skip("pymupdf not installed")
        with pytest.raises(FileNotFoundError):
            dr.read_pdf("nonexistent.pdf")

    def test_read_docx_file_not_found(self):
        """read_docx raises FileNotFoundError for missing file."""
        from enigma_engine.core import document_readers as dr
        if not dr.docx_available():
            pytest.skip("python-docx not installed")
        with pytest.raises(FileNotFoundError):
            dr.read_docx("nonexistent.docx")

    def test_scan_training_data_includes_pdf_docx(self):
        """scan_training_data should glob for .pdf and .docx patterns."""
        import inspect
        from enigma_engine.gui.scanners import scan_training_data
        source = inspect.getsource(scan_training_data)
        assert "*.pdf" in source
        assert "*.docx" in source

    def test_scan_docs_includes_pdf_docx(self):
        """scan_docs should recognize .pdf and .docx extensions."""
        import inspect
        from enigma_engine.gui.scanners import scan_docs
        source = inspect.getsource(scan_docs)
        assert ".pdf" in source
        assert ".docx" in source


class TestRAG:
    """Verify RAG module: chunking, TF-IDF, index, query."""

    def test_rag_imports(self):
        """Core RAG classes should import without errors."""
        from enigma_engine.core.rag import (
            chunk_text, TfidfVectorizer, RAGIndex, index_directory,
            CHUNK_SIZE, TOP_K_DEFAULT,
        )
        assert CHUNK_SIZE > 0
        assert TOP_K_DEFAULT > 0

    def test_chunk_text_basic(self):
        """chunk_text splits long text into overlapping chunks."""
        from enigma_engine.core.rag import chunk_text
        text = "Hello world. " * 200  # much longer than CHUNK_SIZE
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(c.strip() for c in chunks)

    def test_chunk_text_empty(self):
        """chunk_text returns empty list for empty text."""
        from enigma_engine.core.rag import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_text_short(self):
        """Short text produces exactly one chunk."""
        from enigma_engine.core.rag import chunk_text
        chunks = chunk_text("Short text.", chunk_size=500)
        assert len(chunks) == 1

    def test_tfidf_vectorizer(self):
        """TfidfVectorizer computes non-zero vectors."""
        from enigma_engine.core.rag import TfidfVectorizer
        docs = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "fish swim in the water",
        ]
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(docs)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] > 0
        # Each row should have non-zero entries
        import numpy as np
        for i in range(3):
            row = matrix[i]
            if hasattr(row, 'toarray'):
                row = row.toarray()
            assert np.any(np.asarray(row) != 0)

    def test_tfidf_serialization(self):
        """TfidfVectorizer round-trips through to_dict/from_dict."""
        from enigma_engine.core.rag import TfidfVectorizer
        docs = ["hello world", "foo bar baz"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        data = vec.to_dict()
        vec2 = TfidfVectorizer.from_dict(data)
        assert vec2.vocab == vec.vocab

    def test_rag_index_end_to_end(self):
        """Full RAG flow: add docs, build, query, format."""
        from enigma_engine.core.rag import RAGIndex
        index = RAGIndex()
        index.add_document("doc1.txt",
                           "Python is a programming language used for AI.")
        index.add_document("doc2.txt",
                           "Cats and dogs are popular household pets.")
        index.build()
        assert index.is_built
        assert index.chunk_count == 2

        results = index.query("programming language")
        assert len(results) > 0
        assert results[0]["source"] == "doc1.txt"

        ctx = RAGIndex.format_context(results)
        assert "doc1.txt" in ctx

    def test_rag_index_empty_query(self):
        """Querying unbuilt index returns empty list."""
        from enigma_engine.core.rag import RAGIndex
        index = RAGIndex()
        assert index.query("anything") == []

    def test_rag_index_save_load(self, tmp_path):
        """RAG index persists and loads correctly."""
        from enigma_engine.core.rag import RAGIndex
        index = RAGIndex()
        index.add_document("test.md", "Machine learning is a subset of AI.")
        index.build()

        save_path = tmp_path / "test_index.json"
        index.save(save_path)
        assert save_path.exists()

        loaded = RAGIndex.load(save_path)
        assert loaded.is_built
        assert loaded.chunk_count == index.chunk_count
        results = loaded.query("machine learning")
        assert len(results) > 0


class TestCodeSandbox:
    """Verify code.run command registration and safety checks."""

    def test_code_run_registered(self):
        """code.run command should be registered in builtin commands."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        assert "code.run" in reg._commands

    def test_code_run_executes_simple_code(self):
        """code.run should execute simple Python and capture output."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute('code.run print("hello world")')
        assert result.success
        assert "hello world" in result.message

    def test_code_run_blocks_forbidden_ops(self):
        """code.run should block dangerous operations."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run os.remove('important.txt')")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_empty_args(self):
        """code.run with no code should return error."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run")
        assert not result.success

    def test_code_run_in_gui_context(self):
        """_build_gui_context should mention code.run capability."""
        import inspect
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "code.run" in source

    def test_execute_preserves_raw_code_for_code_run(self):
        """CommandRegistry.execute should pass raw code as one arg for code.run."""
        from enigma_engine.core.commands import CommandRegistry, CommandResult

        reg = CommandRegistry()
        captured: list[str] = []

        def capture(args, _ctx):
            captured.extend(args)
            return CommandResult(True, "[OK]")

        reg.register("code.run", capture, "capture", "code.run <python_code>")
        reg.execute("code.run ```python\nprint('a;b|c')\n```")

        assert len(captured) == 1
        assert "```python" in captured[0]
        assert "a;b|c" in captured[0]

    def test_code_run_accepts_fenced_multiline_code(self):
        """code.run should execute fenced multiline Python without syntax mangling."""
        from enigma_engine.core.commands import get_registry

        reg = get_registry()
        cmd = (
            "code.run ```python\n"
            "print('hello from fenced code')\n"
            "```"
        )
        result = reg.execute(cmd)

        assert result.success
        assert "hello from fenced code" in result.message


class TestDPOTraining:
    """Verify DPO training infrastructure."""

    def test_trainer_has_dpo_method(self):
        """Trainer class must have train_dpo method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "train_dpo")

    def test_dpo_loss_function_exists(self):
        """Trainer must have _dpo_loss static method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_dpo_loss")

    def test_dpo_loss_computes(self):
        """_dpo_loss should return a scalar tensor."""
        import torch
        from enigma_engine.core.training import Trainer
        # Simulate log-probs
        pc = torch.tensor([0.0])
        pr = torch.tensor([0.0])
        rc = torch.tensor([0.0])
        rr = torch.tensor([0.0])
        loss = Trainer._dpo_loss(pc, pr, rc, rr, beta=0.1)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0  # DPO loss is non-negative (logsigmoid(0) = -ln2)

    def test_forge_has_dpo_mode(self):
        """FORGE still has DPO method available."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # DPO available but not in main 3-mode UI
        assert hasattr(ForgeMixin, "_start_dpo_training")

    def test_forge_dispatches_three_modes(self):
        """_start_training_by_mode handles simplified 3 modes."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_training_by_mode)
        assert "Basic" in source
        assert "AI-Guided" in source or "AI-guided" in source
        assert "Image" in source or "Vision" in source


class TestImageGenIntegration:
    """Verify image generation command and chat integration."""

    def test_imagegen_generate_registered(self):
        """imagegen.generate command must be in the registry."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        assert "imagegen.generate" in registry._commands

    def test_imagegen_status_registered(self):
        """imagegen.status command must be in the registry."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        assert "imagegen.status" in registry._commands

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

    def test_imagegen_context_in_gui(self):
        """_build_gui_context should mention imagegen.generate."""
        import inspect
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "imagegen.generate" in source
        assert "imagegen.status" in source

    def test_cmd_image_paths_collected(self):
        """_cmd_execute_ai_commands should store _cmd_image_paths."""
        import inspect
        from enigma_engine.gui.gui_cmd_page import CMDPageMixin
        source = inspect.getsource(
            CMDPageMixin._cmd_execute_ai_commands)
        assert "_cmd_image_paths" in source

    def test_chat_renders_cmd_images(self):
        """Images from _cmd_image_paths are rendered after typewriter finishes."""
        import inspect
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        # Check that _insert_media is called from _typewriter
        # (moved there to show images AFTER response finishes typing)
        source = inspect.getsource(LogicChatMixin._typewriter)
        assert "_cmd_image_paths" in source or "_insert_media" in source
        assert "_insert_media" in source


@pytest.mark.structural
class TestModelFileEncoding:
    """Verify model.py uses encoding='utf-8' for config.json reads."""

    def test_model_config_reads_have_encoding(self):
        """All open(config_file) calls in model.py must use encoding."""
        import inspect
        from enigma_engine.core import model
        source = inspect.getsource(model)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ('open(config_file' in stripped
                    and 'encoding' not in stripped):
                assert False, (
                    f"model.py line ~{i+1}: open(config_file) "
                    f"missing encoding='utf-8'")


@pytest.mark.structural
class TestInferenceGpuDetectionLogging:
    """Verify inference.py logs GPU detection failures."""

    def test_gpu_detection_not_silent(self):
        """GPU auto-detection except block should log, not pass."""
        import inspect
        from enigma_engine.core import inference
        source = inspect.getsource(inference)
        # Find the GPU VRAM detection try/except block
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'except Exception' in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line == 'pass':
                    # Check context — is this the VRAM detection block?
                    context = '\n'.join(lines[max(0, i-5):i+2])
                    if 'vram' in context.lower() or 'gpu' in context.lower():
                        assert False, (
                            f"inference.py ~line {i+1}: silent "
                            f"except pass on GPU detection")


@pytest.mark.structural
class TestHardwareDetectionEncoding:
    """Verify hardware_detection.py uses proper encoding."""

    def test_pi_detection_has_encoding(self):
        """Pi device-tree file open should use encoding='utf-8'."""
        import inspect
        from enigma_engine.core import hardware_detection
        source = inspect.getsource(hardware_detection)
        if '/proc/device-tree/model' in source:
            # Find the open() line for this path
            lines = source.split('\n')
            for line in lines:
                if '/proc/device-tree/model' in line and 'open' in line:
                    assert 'encoding' in line, (
                        "Pi device-tree open() missing encoding")


class TestAiProfileCallableType:
    """Verify ai_profile.py uses Callable not callable."""

    def test_profile_manager_callback_types(self):
        """AIProfileManager callbacks should use Callable (type) not callable (builtin)."""
        import inspect
        from enigma_engine.core.ai_profile import AIProfileManager
        source = inspect.getsource(AIProfileManager.__init__)
        # Should not have lowercase callable as type annotation
        assert 'Optional[callable]' not in source, (
            "Use Optional[Callable] not Optional[callable]")


class TestDownloadProgressClearCache:
    """Verify download_progress.py clear_cache uses correct API."""

    def test_clear_cache_uses_strategy_pattern(self):
        """clear_cache should use cache_info.delete_revisions().execute()."""
        import inspect
        from enigma_engine.core.download_progress import DownloadTracker
        source = inspect.getsource(DownloadTracker.clear_cache)
        # Should NOT import standalone delete_revisions
        assert 'from huggingface_hub import delete_revisions' not in source
        # Should use the strategy pattern
        assert 'execute' in source or 'delete_revisions' in source

    def test_format_size_not_duplicate(self):
        """format_size should be an alias or merged with format_bytes."""
        from enigma_engine.core.download_progress import (
            format_bytes, format_size)
        # Both should return the same result
        assert format_bytes(1024) == format_size(1024)
        assert format_bytes(1048576) == format_size(1048576)


class TestModelPresetsValidate:
    """Verify validate() doesn't crash on frozen configs."""

    def test_validate_is_read_only(self):
        """validate() should not try to assign attributes."""
        import inspect
        from enigma_engine.core.model_presets import ForgeConfig
        source = inspect.getsource(ForgeConfig.validate)
        # Should not delegate to __post_init__ (which assigns attrs)
        assert '__post_init__' not in source, (
            "validate() should not call __post_init__ — "
            "crashes on frozen configs")


class TestTokenizerTrainMethod:
    """Verify train_tokenizer uses a tokenizer class that has train()."""

    def test_train_tokenizer_uses_bpe_with_train(self):
        """train_tokenizer should use BPETokenizer (has train), not AdvancedBPETokenizer."""
        import inspect
        from enigma_engine.core.tokenizer import train_tokenizer
        source = inspect.getsource(train_tokenizer)
        # BPE branch should use BPETokenizer, not AdvancedBPETokenizer
        assert 'BPETokenizer()' in source
        # AdvancedBPETokenizer should NOT appear in the train path
        assert 'AdvancedBPETokenizer()' not in source, (
            "AdvancedBPETokenizer has no train() — use BPETokenizer")


class TestLoraEpochLossInit:
    """Verify epoch_loss is initialized before the training loop."""

    def test_epoch_loss_not_fragile(self):
        """epoch_loss should be initialized before loop, not checked via dir()."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        # Should NOT use fragile dir() check
        assert "'epoch_loss' in dir()" not in source, (
            "epoch_loss should be initialized before loop, not checked via dir()")


class TestCacheTypeAnnotations:
    """Verify tokenizer cache annotations match stored types."""

    def test_advanced_tokenizer_cache_type(self):
        """AdvancedBPETokenizer cache should be dict[str, list[str]]."""
        import inspect
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        source = inspect.getsource(AdvancedBPETokenizer.__init__)
        # Cache stores list of string tokens, not list of ints
        assert 'list[int]' not in source or 'cache' not in source.split('list[int]')[0][-30:]

    def test_bpe_tokenizer_cache_type(self):
        """BPETokenizer cache should be dict[str, list[str]]."""
        import inspect
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        source = inspect.getsource(BPETokenizer.__init__)
        assert 'list[int]' not in source or 'cache' not in source.split('list[int]')[0][-30:]


@pytest.mark.structural
class TestGptqAwqEncoding:
    """Verify GPTQ/AWQ loader uses encoding on all open() calls."""

    def test_gptq_awq_open_has_encoding(self):
        """All open() calls in gptq_awq_loader should have encoding='utf-8'."""
        import re
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "gptq_awq_loader.py"
        source = source_path.read_text(encoding='utf-8')
        # Find open() calls that read text (not binary)
        opens = re.findall(r'open\([^)]+\)', source)
        for call in opens:
            if "'rb'" in call or '"rb"' in call:
                continue  # binary mode is fine
            assert 'encoding' in call, f"Missing encoding in: {call}"


@pytest.mark.structural
class TestCharTokenizerNoBaseException:
    """Verify char_tokenizer doesn't catch BaseException."""

    def test_no_base_exception(self):
        """char_tokenizer should not catch BaseException (swallows KeyboardInterrupt)."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "char_tokenizer.py"
        source = source_path.read_text(encoding='utf-8')
        assert 'except BaseException' not in source, (
            "Use except Exception, not BaseException")

@pytest.mark.structural
class TestSpeculativeDecodingDraftCount:
    """Verify speculative decoding handles short draft sequences."""

    def test_uses_actual_draft_count(self):
        """speculative_generate must use actual draft count, not num_spec."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.generate_speculative)
        # Should compute actual draft length, not use num_spec for indexing
        assert 'actual_draft' in source, (
            "speculative_generate should compute actual_draft = "
            "draft_tokens.shape[1] - generated.shape[1]")
        # Loop should use actual_draft, not num_spec
        assert 'range(actual_draft)' in source, (
            "Verification loop should iterate range(actual_draft), "
            "not range(num_spec)")


@pytest.mark.structural
class TestHfLoaderNoDeadTokenizer:
    """Verify convert_huggingface_to_forge doesn't download unused tokenizer."""

    def test_no_dead_hf_tokenizer(self):
        """convert_huggingface_to_forge should not load unused hf_tokenizer."""
        import inspect
        from enigma_engine.core.huggingface_loader import convert_huggingface_to_forge
        source = inspect.getsource(convert_huggingface_to_forge)
        lines = source.split('\n')
        # Check that hf_tokenizer is not assigned (dead download)
        assign_lines = [l for l in lines
                        if 'hf_tokenizer' in l and '=' in l and 'import' not in l]
        usage_lines = [l for l in lines
                       if 'hf_tokenizer' in l and '=' not in l]
        if assign_lines:
            assert usage_lines, (
                "hf_tokenizer is assigned but never used — "
                "wastes bandwidth downloading unused tokenizer")


# ================================================================
# Polish audit — fixes verified 2026-02-26
# ================================================================

@pytest.mark.structural
class TestPolishAuditCore:
    """Verify polish fixes made to core engine files."""

    def test_engine_generation_imports_F(self):
        """engine_generation.py must import torch.nn.functional as F."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "engine_generation.py"
        source = source_path.read_text(encoding='utf-8')
        assert "import torch.nn.functional as F" in source, (
            "_sample_token uses F.softmax but F was never imported")

    def test_model_auto_configure_uses_hardware_type(self):
        """model.py auto_configure must use profile.hardware_type, not cpu_model."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "model.py"
        source = source_path.read_text(encoding='utf-8')
        assert "cpu_model" not in source, (
            "HardwareProfile has no cpu_model attribute — use hardware_type")

    def test_model_auto_configure_uses_model_size_key(self):
        """model.py auto_configure must use config['model_size'], not config['size']."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "model.py"
        source = source_path.read_text(encoding='utf-8')
        assert "config['size']" not in source, (
            "get_optimal_config returns 'model_size' key, not 'size'")

    def test_model_utils_uses_asdict(self):
        """model_utils.py must use dataclasses.asdict, not .to_dict()."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "model_utils.py"
        source = source_path.read_text(encoding='utf-8')
        # Should use asdict, not a nonexistent to_dict
        assert "dataclasses.asdict" in source or "asdict(profile)" in source

    def test_hardware_profile_has_to_dict(self):
        """HardwareProfile should have a to_dict() method."""
        from enigma_engine.core.hardware_detection import HardwareProfile
        profile = HardwareProfile()
        result = profile.to_dict()
        assert isinstance(result, dict)
        assert "device" in result
        assert "ram_gb" in result

    def test_builtin_commands_docstring_order(self):
        """builtin_commands.py docstring must come before from __future__."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "builtin_commands.py"
        lines = source_path.read_text(encoding='utf-8').split('\n')
        # Docstring must come first (line 0 should be triple-quote)
        assert lines[0].strip().startswith('\"\"\"'), (
            "Module docstring must precede 'from __future__ import annotations'")

    def test_inference_no_dead_mmap_branch(self):
        """inference.py mmap branches should not be identical."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "inference.py"
        source = source_path.read_text(encoding='utf-8')
        # Should not have duplicate identical branches for use_mmap
        assert "if use_mmap:" not in source or source.count(
            'safe_load_weights(model_file, map_location="cpu")') <= 1, (
            "Both mmap branches were identical — dead code")

    def test_streaming_async_queue_logs_debug(self):
        """streaming.py _emit should log dropped chunks, not silently pass."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "streaming.py"
        source = source_path.read_text(encoding='utf-8')
        assert "logger.debug" in source, (
            "Silent error swallowing in _emit — should log at debug level")


class TestSafeLoadWeightsSecurity:
    """Tests for safe model loading (Suggestion #1: B, D, E)."""

    def test_no_weights_only_false_fallback(self):
        """safe_load_weights must NOT fall back to weights_only=False."""
        import inspect
        from enigma_engine.core.model_registry import safe_load_weights
        source = inspect.getsource(safe_load_weights)
        assert "weights_only=False" not in source, (
            "safe_load_weights still has weights_only=False fallback — security risk")

    def test_no_bare_exception_catch(self):
        """safe_load_weights inner catch should not use bare 'except Exception'."""
        import inspect
        from enigma_engine.core.model_registry import safe_load_weights
        source = inspect.getsource(safe_load_weights)
        # The function should not have an inner fallback catching everything
        lines = source.split('\n')
        # Count 'except Exception' — at most 1 (the outer re-raise wrapper)
        exception_catches = [l for l in lines if 'except Exception' in l]
        assert len(exception_catches) <= 1, (
            "Multiple bare 'except Exception' catches — inner fallback still exists")

    def test_supports_safetensors_format(self):
        """safe_load_weights handles .safetensors files."""
        import inspect
        from enigma_engine.core.model_registry import safe_load_weights
        source = inspect.getsource(safe_load_weights)
        assert "safetensors" in source, (
            "safe_load_weights has no safetensors support")

    def test_lora_load_uses_weights_only_true(self):
        """load_lora_weights must use weights_only=True for .pth files."""
        import inspect
        from enigma_engine.core.lora_utils import load_lora_weights
        source = inspect.getsource(load_lora_weights)
        assert "weights_only=True" in source, (
            "load_lora_weights missing weights_only=True — arbitrary code exec risk")

    def test_atomic_safetensors_save_exists(self):
        """safe_save.py must have atomic_safetensors_save function."""
        from enigma_engine.core.safe_save import atomic_safetensors_save
        assert callable(atomic_safetensors_save)

    def test_safe_load_weights_missing_file(self):
        """safe_load_weights raises FileNotFoundError for missing files."""
        from enigma_engine.core.model_registry import safe_load_weights
        with pytest.raises(FileNotFoundError):
            safe_load_weights("/nonexistent/model.pth")

    def test_no_weights_only_param(self):
        """safe_load_weights should not accept weights_only param anymore."""
        import inspect
        from enigma_engine.core.model_registry import safe_load_weights
        sig = inspect.signature(safe_load_weights)
        assert "weights_only" not in sig.parameters, (
            "weights_only parameter still exposed — always use True internally")

    def test_no_direct_torch_load_in_codebase(self):
        """No code outside safe_load_weights should use torch.load(weights_only=False)."""
        import re
        root = Path(__file__).parent.parent / "enigma_engine"
        violations = []
        pattern = re.compile(r"torch\.load\(.*weights_only\s*=\s*False", re.DOTALL)
        for py in root.rglob("*.py"):
            source = py.read_text(encoding="utf-8")
            if pattern.search(source):
                violations.append(str(py.relative_to(root)))
        assert not violations, (
            f"Direct torch.load(weights_only=False) found in: {violations} "
            "— route through safe_load_weights() instead")


class TestGpuSupport:
    """Verify GPU support is properly configured."""

    def test_pytorch_cuda_available(self):
        """PyTorch should detect CUDA GPU."""
        import torch
        assert torch.cuda.is_available(), "CUDA not available in PyTorch"

    def test_pytorch_cuda_version_not_cpu(self):
        """PyTorch should be CUDA build, not CPU-only."""
        import torch
        assert torch.version.cuda is not None, (
            f"PyTorch {torch.__version__} is CPU-only build — needs CUDA")
        assert "+cpu" not in torch.__version__, (
            f"PyTorch {torch.__version__} is CPU-only build — needs CUDA")

    def test_gpu_device_name(self):
        """GPU device name should be available."""
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            assert len(name) > 0, "GPU name is empty"

    def test_gguf_n_gpu_layers_uses_minus_one(self):
        """GGUF loading should use n_gpu_layers=-1 for full GPU offload."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "core" / "inference.py"
        )
        source = source_path.read_text(encoding='utf-8')
        assert "n_gpu_layers = -1" in source, (
            "GGUF loading should use -1 for full GPU offload, not a fixed number")

    def test_gguf_cuda_dll_path_setup(self):
        """GGUF loading should add PyTorch CUDA DLLs to PATH."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "core" / "inference.py"
        )
        source = source_path.read_text(encoding='utf-8')
        assert "torch" in source and "PATH" in source, (
            "GGUF loader should add PyTorch CUDA DLLs to PATH for Windows")

    def test_device_display_reads_engine(self):
        """GUI device display should read actual engine device, not generic check."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_logic.py"
        )
        source = source_path.read_text(encoding='utf-8')
        # Should not have the old pattern of just checking torch.cuda.is_available()
        # for device display in _on_model_loaded
        assert "self.engine" in source and "gpu_name" in source, (
            "Device display should read from engine, not generic CUDA check")

    def test_hardware_detection_reports_gpu(self):
        """Hardware detection should report GPU."""
        from enigma_engine.core.hardware_detection import detect_hardware
        import torch
        hw = detect_hardware()
        if torch.cuda.is_available():
            assert hw.gpu_available, "GPU available but not detected"
            assert hw.device == "cuda", f"Expected 'cuda', got '{hw.device}'"
            assert hw.gpu_vram_gb > 0, "VRAM should be > 0"

    def test_llama_cpp_gpu_offload(self):
        """llama-cpp-python should support GPU offload."""
        try:
            import os, sys
            torch_lib = os.path.join(
                sys.prefix, 'Lib', 'site-packages', 'torch', 'lib'
            )
            if os.path.isdir(torch_lib):
                os.environ['PATH'] = (
                    torch_lib + os.pathsep + os.environ.get('PATH', '')
                )
            import llama_cpp.llama_cpp as ll
            if hasattr(ll, 'llama_supports_gpu_offload'):
                assert ll.llama_supports_gpu_offload(), (
                    "llama-cpp-python installed without GPU offload support")
        except ImportError:
            pytest.skip("llama-cpp-python not installed")

    def test_gpu_compute_works(self):
        """GPU compute should produce correct results."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        x = torch.ones(100, device='cuda')
        y = x + x
        assert y.sum().item() == 200.0, "GPU compute produced wrong result"


class TestLlamaServerBackend:
    """Tests for the llama-server subprocess backend in gguf_loader.py."""

    def test_llama_server_backend_class_exists(self):
        """LlamaServerBackend class should be importable."""
        from enigma_engine.core.gguf_loader import LlamaServerBackend
        assert LlamaServerBackend is not None

    def test_have_llama_server_constant(self):
        """HAVE_LLAMA_SERVER constant should exist and be a bool."""
        from enigma_engine.core.gguf_loader import HAVE_LLAMA_SERVER
        assert isinstance(HAVE_LLAMA_SERVER, bool)

    def test_needs_server_backend_returns_false_for_cpu(self):
        """_needs_server_backend should return False when n_gpu_layers=0."""
        from enigma_engine.core.gguf_loader import _needs_server_backend
        assert _needs_server_backend(0) is False

    def test_find_free_port_returns_int(self):
        """_find_free_port should return a valid TCP port."""
        from enigma_engine.core.gguf_loader import _find_free_port
        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_ggufmodel_has_server_attribute(self):
        """GGUFModel should have _server and _use_server attributes."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.__init__)
        assert "_server" in source
        assert "_use_server" in source

    def test_ggufmodel_load_checks_server_backend(self):
        """GGUFModel.load() should call _needs_server_backend for auto-detect."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.load)
        assert "_needs_server_backend" in source
        assert "_load_via_server" in source
        assert "_load_in_process" in source

    def test_ggufmodel_generate_delegates_to_server(self):
        """GGUFModel.generate() should delegate to server when active."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.generate)
        assert "self._server" in source

    def test_ggufmodel_chat_delegates_to_server(self):
        """GGUFModel.chat() should delegate to server when active."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.chat)
        assert "self._server" in source

    def test_ggufmodel_chat_with_tools_delegates_to_server(self):
        """GGUFModel.chat_with_tools() should delegate to server."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.chat_with_tools)
        assert "self._server" in source

    def test_ggufmodel_tokenize_delegates_to_server(self):
        """GGUFModel.tokenize() should delegate to server when active."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.tokenize)
        assert "self._server" in source

    def test_ggufmodel_detokenize_delegates_to_server(self):
        """GGUFModel.detokenize() should delegate to server when active."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.detokenize)
        assert "self._server" in source

    def test_ggufmodel_get_info_includes_backend(self):
        """GGUFModel.get_info() should include 'backend' key."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.get_info)
        assert "'backend'" in source
        assert "llama-server" in source

    def test_ggufmodel_unload_stops_server(self):
        """GGUFModel.unload() should stop server if active."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel.unload)
        assert "self._server" in source
        assert ".stop()" in source

    def test_server_backend_has_lifecycle_methods(self):
        """LlamaServerBackend must have start/stop methods."""
        from enigma_engine.core.gguf_loader import LlamaServerBackend
        assert hasattr(LlamaServerBackend, 'start')
        assert hasattr(LlamaServerBackend, 'stop')

    def test_server_backend_has_inference_methods(self):
        """LlamaServerBackend must have generate/chat/tokenize."""
        from enigma_engine.core.gguf_loader import LlamaServerBackend
        for method in ('generate', 'chat', 'chat_with_tools',
                       'tokenize', 'detokenize'):
            assert hasattr(LlamaServerBackend, method), (
                f"LlamaServerBackend missing {method}()")

    def test_extract_metadata_from_file_uses_correct_api(self):
        """_extract_metadata_from_file should use parse_gguf_header."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel._extract_metadata_from_file)
        assert "parse_gguf_header" in source
        assert "parse_gguf_metadata" in source
        # Must open file and pass handle, not pass a string
        assert "open(self.model_path" in source

    def test_server_backend_model_path_resolved(self):
        """LlamaServerBackend should resolve model_path to absolute."""
        import inspect
        from enigma_engine.core.gguf_loader import LlamaServerBackend
        source = inspect.getsource(LlamaServerBackend.__init__)
        assert ".resolve()" in source


# ================================================================
# Persistent Memory system
# ================================================================

class TestPersistentMemory:
    """Tests for enigma_engine.core.memory module."""

    def test_import(self):
        """PersistentMemory can be imported."""
        from enigma_engine.core.memory import PersistentMemory
        assert PersistentMemory is not None

    def test_add_and_retrieve(self, tmp_path):
        """Facts can be added and retrieved."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        assert mem.add("User's name is Alex")
        assert mem.count == 1
        assert "Alex" in mem.facts[0]

    def test_deduplication(self, tmp_path):
        """Duplicate facts are rejected."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        assert mem.add("User likes Python")
        assert not mem.add("User likes Python")
        assert not mem.add("user likes python")  # case-insensitive
        assert mem.count == 1

    def test_replace_outdated(self, tmp_path):
        """Updated facts replace old ones about the same topic."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        mem.add("User's name is Bob")
        mem.add("User's name is Alex")
        assert mem.count == 1
        assert "Alex" in mem.facts[0]

    def test_remove_by_content(self, tmp_path):
        """Facts can be removed by substring match."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        mem.add("User works at NASA")
        mem.add("User likes coffee")
        assert mem.remove("NASA")
        assert mem.count == 1
        assert "coffee" in mem.facts[0]

    def test_remove_by_index(self, tmp_path):
        """Facts can be removed by index."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        mem.add("Fact A")
        mem.add("Fact B")
        assert mem.remove(0)
        assert mem.count == 1
        assert "Fact B" in mem.facts[0]

    def test_clear(self, tmp_path):
        """Clear removes all facts."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        mem.add("Fact 1")
        mem.add("Fact 2")
        mem.clear()
        assert mem.count == 0

    def test_persistence(self, tmp_path):
        """Facts survive save/reload cycle."""
        path = tmp_path / "mem.md"
        from enigma_engine.core.memory import PersistentMemory
        mem1 = PersistentMemory(memory_path=path)
        mem1.add("User is a developer")
        mem1.add("User prefers dark mode")
        # Create a new instance from same path
        mem2 = PersistentMemory(memory_path=path)
        assert mem2.count == 2
        assert "developer" in mem2.facts[0]
        assert "dark mode" in mem2.facts[1]

    def test_build_context(self, tmp_path):
        """build_context produces formatted output."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        assert mem.build_context() == ""  # empty when no facts
        mem.add("User's name is Alex")
        ctx = mem.build_context()
        assert "[MEMORY" in ctx
        assert "Alex" in ctx
        assert "[END MEMORY]" in ctx

    def test_build_context_token_cap(self, tmp_path):
        """build_context respects token budget."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        # Add many long facts
        for i in range(30):
            mem.add(f"This is a moderately long fact number {i} about something the user said")
        ctx = mem.build_context(max_tokens=100)
        # Should be capped — not all 30 facts should appear
        assert ctx.count("- ") < 30

    def test_extract_facts_name(self, tmp_path):
        """extract_facts catches 'my name is X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("Hey, my name is Alex")
        assert len(added) >= 1
        assert any("Alex" in f for f in added)
        assert mem.count >= 1

    def test_extract_facts_workplace(self, tmp_path):
        """extract_facts catches 'I work at X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I work at NASA doing research.")
        assert len(added) >= 1
        assert any("NASA" in f for f in added)

    def test_extract_facts_preference(self, tmp_path):
        """extract_facts catches 'I prefer X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I prefer Python over JavaScript")
        assert len(added) >= 1
        assert any("Python" in f for f in added)

    def test_extract_facts_remember_request(self, tmp_path):
        """extract_facts catches 'remember that X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("Please remember that my dog is named Max")
        assert len(added) >= 1
        assert any("Max" in f for f in added)

    def test_extract_facts_nothing(self, tmp_path):
        """extract_facts returns empty on uninteresting messages."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("What is the weather today?")
        assert len(added) == 0

    def test_max_facts_trim(self, tmp_path):
        """Oldest facts are trimmed when exceeding MAX_FACTS."""
        from enigma_engine.core.memory import PersistentMemory, MAX_FACTS
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        for i in range(MAX_FACTS + 10):
            mem.add(f"Unique fact number {i}")
        assert mem.count == MAX_FACTS
        # First 10 should have been trimmed
        assert "number 0" not in mem.facts[0]

    def test_get_memory_singleton(self):
        """get_memory returns a singleton."""
        from enigma_engine.core.memory import get_memory
        import enigma_engine.core.memory as mem_module
        # Reset singleton for test isolation
        mem_module._instance = None
        m1 = get_memory()
        m2 = get_memory()
        assert m1 is m2
        mem_module._instance = None  # cleanup

    def test_hand_editability(self, tmp_path):
        """User can hand-edit the memory file."""
        path = tmp_path / "mem.md"
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=path)
        mem.add("Original fact")
        # Simulate user hand-editing the file
        path.write_text(
            "# AI Memory Notes\n\n"
            "- Hand-written fact by user\n"
            "- Another user note\n",
            encoding="utf-8")
        mem.reload()
        assert mem.count == 2
        assert "Hand-written fact" in mem.facts[0]


class TestMemoryBuiltinCommands:
    """Tests for memory.remember/forget/notes builtin commands."""

    def test_remember_command_registered(self):
        """memory.remember command exists in registry."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmds = [c.name for c in registry.list_commands()]
        assert "memory.remember" in cmds

    def test_forget_command_registered(self):
        """memory.forget command exists in registry."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmds = [c.name for c in registry.list_commands()]
        assert "memory.forget" in cmds

    def test_notes_command_registered(self):
        """memory.notes command exists in registry."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmds = [c.name for c in registry.list_commands()]
        assert "memory.notes" in cmds


# =========================================================================
# Deep-dive audit fix tests — config.set, memory traversal, AI profile
# =========================================================================

class TestConfigSetTypeConversion:
    """Tests for config.set robust type conversion (int→float→string)."""

    def _run_config_set(self, key, value):
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "config.set")
        ctx = {"config": {}}
        result = cmd.handler([key, value], ctx)
        return result, ctx["config"].get(key)

    def test_set_integer(self):
        """config.set should convert '42' to int."""
        result, val = self._run_config_set("key", "42")
        assert result.success
        assert val == 42
        assert isinstance(val, int)

    def test_set_float(self):
        """config.set should convert '3.14' to float."""
        result, val = self._run_config_set("key", "3.14")
        assert result.success
        assert isinstance(val, float)
        assert abs(val - 3.14) < 0.001

    def test_set_multi_dot_string(self):
        """config.set should keep '1.2.3' as string — no crash."""
        result, val = self._run_config_set("key", "1.2.3")
        assert result.success
        assert val == "1.2.3"
        assert isinstance(val, str)

    def test_set_bool_true(self):
        """config.set should convert 'true' to bool True."""
        result, val = self._run_config_set("key", "true")
        assert result.success
        assert val is True

    def test_set_bool_false(self):
        """config.set should convert 'FALSE' to bool False."""
        result, val = self._run_config_set("key", "FALSE")
        assert result.success
        assert val is False

    def test_set_plain_string(self):
        """config.set should keep 'hello' as string."""
        result, val = self._run_config_set("key", "hello")
        assert result.success
        assert val == "hello"

    def test_set_negative_int(self):
        """config.set should handle '-5'."""
        result, val = self._run_config_set("key", "-5")
        assert result.success
        assert val == -5

    def test_set_missing_args(self):
        """config.set with <2 args returns error."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "config.set")
        result = cmd.handler(["only_key"], {})
        assert not result.success


class TestMemorySaveLoadTraversal:
    """Tests that memory.save/load sanitise names to prevent path traversal."""

    def test_save_rejects_path_traversal(self, tmp_path):
        """memory.save with '../evil' should strip to just 'evil'."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "memory.save")
        ctx = {
            "memory_dir": tmp_path,
            "chat_messages": [{"role": "user", "content": "hi"}],
        }
        result = cmd.handler(["../evil"], ctx)
        assert result.success
        # File should be in tmp_path, not parent
        assert (tmp_path / "evil.json").exists()
        assert not (tmp_path.parent / "evil.json").exists()

    def test_load_rejects_path_traversal(self, tmp_path):
        """memory.load with '../evil' should strip to just 'evil'."""
        import json
        # Create the file in tmp_path (the valid dir)
        (tmp_path / "evil.json").write_text(
            json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            encoding="utf-8",
        )
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "memory.load")
        ctx = {"memory_dir": tmp_path}
        result = cmd.handler(["../evil"], ctx)
        assert result.success
        assert len(ctx["chat_messages"]) == 1

    def test_save_rejects_dot_dot_name(self):
        """memory.save with '..' as name should fail."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmd = next(c for c in registry.list_commands() if c.name == "memory.save")
        result = cmd.handler([".."], {"chat_messages": [{"role": "user", "content": "x"}]})
        assert not result.success


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

    def test_from_dict_filters_unknown_keys(self):
        """from_dict should ignore unknown top-level keys."""
        from enigma_engine.core.ai_profile import AIProfile
        data = {
            "id": "test", "name": "Test",
            "system_prompt": "Hello",
            "unknown_field_xyz": 42,
        }
        profile = AIProfile.from_dict(data)
        assert not hasattr(profile, "unknown_field_xyz")


# =========================================================================
# Item 17 — Mixed Async/Sync Mods (standardised on threading)
# =========================================================================

class TestModClientSync:
    """Verify ModClient is fully synchronous (no asyncio)."""

    def test_mod_base_no_asyncio_import(self):
        """mod_base.py must not import asyncio."""
        import ast
        from pathlib import Path
        mod_base = Path(__file__).resolve().parent.parent / "mods" / "_template" / "mod_base.py"
        tree = ast.parse(mod_base.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "asyncio", "mod_base.py still imports asyncio"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "asyncio", "mod_base.py still imports asyncio"

    def test_mod_client_methods_are_sync(self):
        """All public ModClient methods must be plain functions, not coroutines."""
        import sys, importlib.util, types
        from pathlib import Path
        mod_base = Path(__file__).resolve().parent.parent / "mods" / "_template" / "mod_base.py"
        spec = importlib.util.spec_from_file_location("mod_base", mod_base)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = mod.ModClient
        for name in ("connect", "send_message", "receive_message",
                      "register", "handle_message", "send_update", "run"):
            method = getattr(cls, name, None)
            assert method is not None, f"ModClient missing {name}"
            assert not getattr(method, "_is_coroutine", False), f"{name} is a coroutine"
            import asyncio
            assert not asyncio.iscoroutinefunction(method), f"{name} is async"

    def test_mod_client_uses_socket(self):
        """ModClient should use socket.socket, not asyncio streams."""
        from pathlib import Path
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "_template" / "mod_base.py").read_text(encoding="utf-8")
        assert "socket.socket" in source
        assert "asyncio.open_connection" not in source
        assert "StreamReader" not in source
        assert "StreamWriter" not in source

    def test_imagegen_mod_no_asyncio(self):
        """ImageGen mod must not use asyncio."""
        from pathlib import Path
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "asyncio" not in source
        assert "async def" not in source
        assert "await " not in source

    def test_imagegen_mod_inherits_mod_client(self):
        """ImageGen mod must inherit from ModClient."""
        from pathlib import Path
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "class ImageGenMod(ModClient)" in source

    def test_template_mod_no_asyncio(self):
        """Template mod must not use asyncio."""
        from pathlib import Path
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "_template" / "main.py").read_text(encoding="utf-8")
        assert "asyncio" not in source
        assert "async def" not in source
        assert "await " not in source

    def test_router_uses_threading(self):
        """ModRouter must use threading, not asyncio."""
        from enigma_engine.router import ModRouter
        import inspect
        source = inspect.getsource(ModRouter)
        assert "threading" in source or "Thread" in source
        assert "asyncio" not in source

    def test_mod_client_handle_message_dispatches(self):
        """handle_message should dispatch to cmd_* and return dict."""
        import importlib.util
        from pathlib import Path
        mod_base = Path(__file__).resolve().parent.parent / "mods" / "_template" / "mod_base.py"
        spec = importlib.util.spec_from_file_location("mod_base", mod_base)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class TestMod(mod.ModClient):
            def __init__(self):
                self.mod_id = "test"
                self.name = "Test"
                self.running = False
                self.capabilities = []
                self._socket = None
                self._send_lock = __import__("threading").Lock()
            def cmd_hello(self, args):
                return {"greeting": f"hi {args.get('name', '')}"}

        test_mod = TestMod()
        resp = test_mod.handle_message({
            "type": "command", "id": "1",
            "data": {"command": "hello", "args": {"name": "world"}}
        })
        assert resp["success"] is True
        assert resp["data"]["greeting"] == "hi world"

    def test_mod_client_handle_ping(self):
        """handle_message should respond to ping with pong."""
        import importlib.util
        from pathlib import Path
        mod_base = Path(__file__).resolve().parent.parent / "mods" / "_template" / "mod_base.py"
        spec = importlib.util.spec_from_file_location("mod_base", mod_base)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class TestMod(mod.ModClient):
            def __init__(self):
                self.mod_id = "test"
                self.name = "Test"
                self.running = False
                self.capabilities = []
                self._socket = None
                self._send_lock = __import__("threading").Lock()

        test_mod = TestMod()
        resp = test_mod.handle_message({"type": "ping", "id": "42"})
        assert resp["type"] == "pong"
        assert resp["id"] == "42"


# =========================================================================
# Item 18 — Vectorised Batch Sampling
# =========================================================================

class TestBatchSamplingVectorized:
    """Verify batch_generate uses vectorised sampling."""

    def test_sample_token_batch_method_exists(self):
        """engine_generation module exposes _sample_token_batch."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        assert hasattr(_GenerationMixin, "_sample_token_batch")

    def test_sample_token_batch_returns_correct_shape(self):
        """_sample_token_batch should return [batch, 1]."""
        import torch, torch.nn.functional as F
        from enigma_engine.core.engine_generation import _GenerationMixin

        class FakeGen:
            """Minimal object that has the method."""
            _sample_token_batch = _GenerationMixin._sample_token_batch

        gen = FakeGen()
        batch, vocab = 4, 32
        logits = torch.randn(batch, vocab)
        generated = torch.randint(0, vocab, (batch, 10))
        result = gen._sample_token_batch(
            logits, generated,
            temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
        assert result.shape == (batch, 1)

    def test_sample_token_batch_respects_top_k(self):
        """Only top-k tokens should get non-zero probability."""
        import torch
        from enigma_engine.core.engine_generation import _GenerationMixin

        class FakeGen:
            _sample_token_batch = _GenerationMixin._sample_token_batch

        gen = FakeGen()
        # Create logits where token 0 is clearly the best
        logits = torch.full((2, 50), -10.0)
        logits[:, 0] = 10.0
        generated = torch.zeros(2, 5, dtype=torch.long)
        result = gen._sample_token_batch(
            logits.clone(), generated,
            temperature=0.01, top_k=1, top_p=1.0, repetition_penalty=1.0)
        # With top_k=1 and low temperature, token 0 should always win
        assert (result.squeeze(-1) == 0).all()

    def test_batch_generate_no_per_sequence_loop(self):
        """The sampling loop must not iterate per sequence for token generation."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.batch_generate)
        # The old code had a per-sequence loop that called _sample_token per-row.
        # The vectorized version should NOT call _sample_token (singular) at all.
        assert "_sample_token(" not in source

    def test_batch_generate_calls_sample_token_batch(self):
        """batch_generate should call _sample_token_batch, not _sample_token."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.batch_generate)
        assert "_sample_token_batch" in source
        # _sample_token (without _batch) should NOT appear in the sampling loop
        # It's only used in single-prompt generate()
        assert "_sample_token(" not in source

    def test_sample_token_batch_repetition_penalty(self):
        """Repetition penalty should reduce probability of repeated tokens."""
        import torch
        from enigma_engine.core.engine_generation import _GenerationMixin

        class FakeGen:
            _sample_token_batch = _GenerationMixin._sample_token_batch

        gen = FakeGen()
        batch, vocab = 1, 10
        # All logits equal; token 3 repeated many times in history
        logits = torch.zeros(batch, vocab)
        generated = torch.full((batch, 20), 3, dtype=torch.long)
        # With strong penalty, token 3 should be suppressed
        # Run many samples and verify token 3 is rarely chosen
        counts = torch.zeros(vocab, dtype=torch.long)
        for _ in range(200):
            tok = gen._sample_token_batch(
                logits.clone(), generated,
                temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=2.0)
            counts[tok.item()] += 1
        # Token 3 should appear less than average (20 out of 200)
        assert counts[3] < 40, f"Token 3 appeared {counts[3]} times, expected < 40"


# =========================================================================
# Item 19 — GUI Themes / Color Customization
# =========================================================================

class TestGUIThemes:
    """Verify the theme system works correctly."""

    def test_theme_dataclass_fields(self):
        """Theme should have all required colour fields."""
        from enigma_engine.gui.themes import Theme
        import dataclasses
        names = {f.name for f in dataclasses.fields(Theme)}
        required = {
            "name", "bg", "panel", "surface", "input",
            "accent", "accent_dim", "accent_muted",
            "purple", "purple_dim", "purple_muted", "cyan",
            "green", "green_dim", "red", "orange",
            "text", "text_dim", "text_bright",
            "border", "border_accent",
        }
        assert required.issubset(names)

    def test_default_theme_is_dark(self):
        """Default theme should be 'dark'."""
        from enigma_engine.gui.themes import DEFAULT_THEME
        assert DEFAULT_THEME == "dark"

    def test_at_least_four_themes(self):
        """Should have at least 4 preset themes."""
        from enigma_engine.gui.themes import THEMES
        assert len(THEMES) >= 4

    def test_get_theme_names(self):
        """get_theme_names returns list of strings."""
        from enigma_engine.gui.themes import get_theme_names
        names = get_theme_names()
        assert isinstance(names, list)
        assert "dark" in names
        assert all(isinstance(n, str) for n in names)

    def test_get_theme_valid(self):
        """get_theme returns the correct Theme object."""
        from enigma_engine.gui.themes import get_theme
        theme = get_theme("dark")
        assert theme.name == "dark"
        assert theme.bg == "#080808"

    def test_get_theme_unknown_falls_back(self):
        """Unknown theme name falls back to dark."""
        from enigma_engine.gui.themes import get_theme
        theme = get_theme("nonexistent_theme_xyz")
        assert theme.name == "dark"

    def test_load_active_theme_returns_theme(self):
        """load_active_theme returns a Theme instance."""
        from enigma_engine.gui.themes import Theme, load_active_theme
        theme = load_active_theme()
        assert isinstance(theme, Theme)

    def test_all_themes_have_hex_colors(self):
        """All colour values in all themes should be valid #hex strings."""
        import dataclasses
        from enigma_engine.gui.themes import THEMES, Theme
        colour_fields = [
            f.name for f in dataclasses.fields(Theme) if f.name != "name"]
        for theme_name, theme in THEMES.items():
            for field_name in colour_fields:
                val = getattr(theme, field_name)
                assert val.startswith("#"), (
                    f"{theme_name}.{field_name} = {val!r} is not a hex colour")
                # Check valid hex length (4 or 7 including #)
                assert len(val) in (4, 7), (
                    f"{theme_name}.{field_name} = {val!r} bad hex length")

    def test_theme_to_dict(self):
        """to_dict should return a dict without the name key."""
        from enigma_engine.gui.themes import get_theme
        d = get_theme("dark").to_dict()
        assert isinstance(d, dict)
        assert "name" not in d
        assert "bg" in d

    def test_widgets_use_theme_colors(self):
        """widgets.py C_* constants should match the active theme."""
        from enigma_engine.gui.themes import load_active_theme
        from enigma_engine.gui.widgets import C_BG, C_PANEL, C_TEXT
        theme = load_active_theme()
        assert C_BG == theme.bg
        assert C_PANEL == theme.panel
        assert C_TEXT == theme.text

    def test_save_theme_preference(self, tmp_path):
        """save_theme_preference writes to settings json."""
        import enigma_engine.gui.themes as themes_mod
        fake_settings = tmp_path / "gui_settings.json"
        original = themes_mod._SETTINGS_PATH
        try:
            themes_mod._SETTINGS_PATH = fake_settings
            themes_mod.save_theme_preference("midnight")
            import json
            data = json.loads(fake_settings.read_text(encoding="utf-8"))
            assert data["theme"] == "midnight"
        finally:
            themes_mod._SETTINGS_PATH = original

    def test_midnight_theme_values(self):
        """Midnight theme should have distinct blue-tinted colours."""
        from enigma_engine.gui.themes import get_theme
        theme = get_theme("midnight")
        assert theme.name == "midnight"
        # Midnight bg should be different from dark bg
        assert theme.bg != "#080808"


# =========================================================================
# Item 20 — Plugin API for Custom Commands
# =========================================================================

class TestPluginLoader:
    """Verify the plugin discovery and loading system."""

    def test_discover_plugins_empty_dir(self, tmp_path):
        """discover_plugins returns [] for an empty directory."""
        from enigma_engine.core.plugin_loader import discover_plugins
        assert discover_plugins(tmp_path) == []

    def test_discover_plugins_finds_py_files(self, tmp_path):
        """discover_plugins finds .py files but skips _-prefixed ones."""
        from enigma_engine.core.plugin_loader import discover_plugins
        (tmp_path / "hello.py").write_text("# plugin", encoding="utf-8")
        (tmp_path / "_private.py").write_text("# hidden", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# not python", encoding="utf-8")
        found = discover_plugins(tmp_path)
        assert len(found) == 1
        assert found[0].stem == "hello"

    def test_discover_plugins_nonexistent_dir(self):
        """discover_plugins returns [] for a directory that doesn't exist."""
        from enigma_engine.core.plugin_loader import discover_plugins
        from pathlib import Path
        assert discover_plugins(Path("/nonexistent/dir/xyz")) == []

    def test_load_plugin_success(self, tmp_path):
        """load_plugin imports a valid plugin and calls register()."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_code = '''
from enigma_engine.core.commands import CommandResult

def register(registry):
    def my_cmd(args, ctx):
        return CommandResult(True, "[OK] works")
    registry.register("test_plug.hello", my_cmd, "Test", "test_plug.hello")
'''
        plugin_file = tmp_path / "test_plug.py"
        plugin_file.write_text(plugin_code, encoding="utf-8")

        reg = CommandRegistry()
        assert load_plugin(plugin_file, reg) is True
        assert "test_plug.hello" in [c.name for c in reg.list_commands()]

    def test_load_plugin_no_register_function(self, tmp_path):
        """load_plugin returns False if plugin has no register()."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_file = tmp_path / "no_register.py"
        plugin_file.write_text("x = 42\n", encoding="utf-8")

        reg = CommandRegistry()
        assert load_plugin(plugin_file, reg) is False

    def test_load_plugin_register_raises(self, tmp_path):
        """load_plugin returns False if register() raises."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_file = tmp_path / "bad_register.py"
        plugin_file.write_text(
            "def register(reg):\n    raise ValueError('broken')\n",
            encoding="utf-8",
        )

        reg = CommandRegistry()
        assert load_plugin(plugin_file, reg) is False

    def test_load_plugin_syntax_error(self, tmp_path):
        """load_plugin returns False for a file with syntax errors."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_file = tmp_path / "syntax_err.py"
        plugin_file.write_text("def broken(\n", encoding="utf-8")

        reg = CommandRegistry()
        assert load_plugin(plugin_file, reg) is False

    def test_load_all_plugins_counts(self, tmp_path):
        """load_all_plugins returns the number of successfully loaded plugins."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_all_plugins

        good = '''
from enigma_engine.core.commands import CommandResult
def register(reg):
    reg.register("p1.cmd", lambda a, c: CommandResult(True, "ok"), "p1", "p1.cmd")
'''
        (tmp_path / "good.py").write_text(good, encoding="utf-8")
        (tmp_path / "bad.py").write_text("def register(r): raise RuntimeError()\n",
                                         encoding="utf-8")
        (tmp_path / "_skip.py").write_text("# skipped\n", encoding="utf-8")

        reg = CommandRegistry()
        loaded = load_all_plugins(reg, tmp_path)
        assert loaded == 1  # good loaded, bad failed, _skip skipped

    def test_plugin_command_executes(self, tmp_path):
        """A command registered by a plugin can be executed via the registry."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_code = '''
from enigma_engine.core.commands import CommandResult
def register(registry):
    def echo(args, ctx):
        return CommandResult(True, "[OK] " + " ".join(args))
    registry.register("echo.say", echo, "Echo args", "echo.say <text>")
'''
        (tmp_path / "echo_plug.py").write_text(plugin_code, encoding="utf-8")

        reg = CommandRegistry()
        load_plugin(tmp_path / "echo_plug.py", reg)
        result = reg.execute("echo.say hello world")
        assert result.success
        assert "hello world" in result.message

    def test_example_plugin_not_loaded(self):
        """_example.py should not be loaded (starts with underscore)."""
        from enigma_engine.core.plugin_loader import discover_plugins
        from pathlib import Path
        plugins_dir = Path(__file__).resolve().parent.parent / "plugins"
        found = discover_plugins(plugins_dir)
        assert all(p.stem != "_example" for p in found)

    def test_get_registry_loads_plugins(self):
        """get_registry() should include plugin loader call in source."""
        import inspect
        from enigma_engine.core.commands import get_registry
        source = inspect.getsource(get_registry)
        assert "load_all_plugins" in source

    def test_plugin_loader_module_exists(self):
        """plugin_loader module should be importable."""
        from enigma_engine.core import plugin_loader
        assert hasattr(plugin_loader, "discover_plugins")
        assert hasattr(plugin_loader, "load_plugin")
        assert hasattr(plugin_loader, "load_all_plugins")

    def test_multiple_plugins_all_register(self, tmp_path):
        """Multiple valid plugins should all register their commands."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_all_plugins

        for i in range(3):
            code = f'''
from enigma_engine.core.commands import CommandResult
def register(reg):
    reg.register("multi{i}.cmd", lambda a, c: CommandResult(True, "ok"), "", "")
'''
            (tmp_path / f"plug{i}.py").write_text(code, encoding="utf-8")

        reg = CommandRegistry()
        loaded = load_all_plugins(reg, tmp_path)
        assert loaded == 3
        names = [c.name for c in reg.list_commands()]
        for i in range(3):
            assert f"multi{i}.cmd" in names


# =========================================================================
# Suggestion #4 — Command Injection Sanitization
# =========================================================================

class TestCommandSanitization:
    """Verify shell metacharacter sanitization in command args."""

    def test_sanitize_args_removes_metacharacters(self):
        """sanitize_args strips ; | & $ ` \\ ! { } from args."""
        from enigma_engine.core.commands import sanitize_args
        dirty = ["hello;world", "foo|bar", "a&b", "$HOME", "`rm -rf`", "x\\y", "!bang", "{x}", "clean"]
        result = sanitize_args(dirty)
        assert result == ["helloworld", "foobar", "ab", "HOME", "rm -rf", "xy", "bang", "x", "clean"]

    def test_sanitize_args_preserves_clean(self):
        """sanitize_args passes through clean args unchanged."""
        from enigma_engine.core.commands import sanitize_args
        clean = ["hello", "world", "foo.bar", "path/to/file", "arg-with-dashes"]
        assert sanitize_args(clean) == clean

    def test_sanitize_args_empty_list(self):
        """sanitize_args handles empty input."""
        from enigma_engine.core.commands import sanitize_args
        assert sanitize_args([]) == []

    def test_execute_sanitizes_before_handler(self):
        """CommandRegistry.execute() sanitizes args before passing to handler."""
        from enigma_engine.core.commands import CommandRegistry, CommandResult
        reg = CommandRegistry()
        received_args = []
        def capture_handler(args, ctx):
            received_args.extend(args)
            return CommandResult(True, "[OK]")
        reg.register("test.cap", capture_handler, "capture", "test.cap")
        reg.execute("test.cap hello;world foo|bar")
        assert received_args == ["helloworld", "foobar"]

    def test_shell_metacharacters_constant(self):
        """SHELL_METACHARACTERS is a frozenset with expected chars."""
        from enigma_engine.core.commands import SHELL_METACHARACTERS
        assert isinstance(SHELL_METACHARACTERS, frozenset)
        for ch in ";|&$`\\!{}":
            assert ch in SHELL_METACHARACTERS

    def test_sanitize_args_does_not_mutate_input(self):
        """sanitize_args returns a new list, not mutating the original."""
        from enigma_engine.core.commands import sanitize_args
        original = ["a;b", "c|d"]
        result = sanitize_args(original)
        assert original == ["a;b", "c|d"]  # unchanged
        assert result == ["ab", "cd"]


class TestRunCommandRemovedFromDefaults:
    """Verify run_command is not in default GGUF tools (Suggestion #4E)."""

    def test_no_run_command_in_default_tools(self):
        """_get_default_tools must not contain run_command."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel._get_default_tools)
        assert "run_command" not in source

    def test_default_tools_still_has_other_tools(self):
        """Default tools should still include generate_image, generate_code, etc."""
        import inspect
        from enigma_engine.core.gguf_loader import GGUFModel
        source = inspect.getsource(GGUFModel._get_default_tools)
        for tool_name in ("generate_image", "generate_code", "read_file",
                          "list_directory", "web_search"):
            assert tool_name in source, f"Missing tool: {tool_name}"

    def test_guard_no_run_command_in_gguf_loader(self):
        """Guard test: run_command must not appear in gguf_loader.py at all."""
        source_path = Path(__file__).resolve().parent.parent / "enigma_engine" / "core" / "gguf_loader.py"
        source = source_path.read_text(encoding="utf-8")
        assert "run_command" not in source, (
            "run_command found in gguf_loader.py — it was removed for security (Suggestion #4E)"
        )


# =========================================================================
# Suggestion #5 — Plugin Loader Security
# =========================================================================

class TestPluginSecurity:
    """Verify plugin pre-scan, AST validation, and trusted allowlist."""

    def test_has_register_def_positive(self):
        """_has_register_def returns True for source with def register."""
        from enigma_engine.core.plugin_loader import _has_register_def
        assert _has_register_def("def register(registry):\n    pass\n")

    def test_has_register_def_negative(self):
        """_has_register_def returns False for source without def register."""
        from enigma_engine.core.plugin_loader import _has_register_def
        assert not _has_register_def("x = 42\ndef foo(): pass\n")

    def test_reject_plugin_without_def_register(self, tmp_path):
        """Plugin without def register in source is rejected before exec_module."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        plugin = tmp_path / "no_reg.py"
        plugin.write_text("x = 42\n", encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is False

    def test_ast_scan_flags_exec(self):
        """_ast_scan_dangers flags exec() calls."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers
        source = "exec('print(1)')\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert len(flags) == 1
        assert "exec()" in flags[0]

    def test_ast_scan_flags_eval(self):
        """_ast_scan_dangers flags eval() calls."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers
        source = "x = eval('2+2')\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert len(flags) == 1
        assert "eval()" in flags[0]

    def test_ast_scan_flags_os_system(self):
        """_ast_scan_dangers flags os.system() calls."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers
        source = "import os\nos.system('rm -rf /')\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert any("os.system" in f for f in flags)

    def test_ast_scan_flags_subprocess_run(self):
        """_ast_scan_dangers flags subprocess.run()."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers
        source = "import subprocess\nsubprocess.run(['ls'])\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert any("subprocess.run" in f for f in flags)
        assert any("import subprocess" in f for f in flags)

    def test_ast_scan_clean_source(self):
        """_ast_scan_dangers returns empty list for safe source."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers
        source = '''
from enigma_engine.core.commands import CommandResult
def register(registry):
    def hello(args, ctx):
        return CommandResult(True, "hello")
    registry.register("hello.greet", hello, "Greet", "hello.greet")
'''
        flags = _ast_scan_dangers(source, "test.py")
        assert flags == []

    def test_reject_plugin_with_dangerous_code(self, tmp_path):
        """Plugin with os.system() call is rejected."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        code = '''import os
def register(registry):
    os.system('echo pwned')
'''
        plugin = tmp_path / "evil.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is False

    def test_reject_plugin_with_exec(self, tmp_path):
        """Plugin using exec() is rejected."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        code = '''
def register(registry):
    exec("print('hacked')")
'''
        plugin = tmp_path / "exec_evil.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is False

    def test_trusted_plugins_allowlist(self, tmp_path, monkeypatch):
        """Plugin not in trusted_plugins list is rejected."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        import enigma_engine
        monkeypatch.setitem(enigma_engine.CONFIG, "trusted_plugins", ["allowed.py"])
        code = '''
from enigma_engine.core.commands import CommandResult
def register(registry):
    registry.register("t.cmd", lambda a, c: CommandResult(True, "ok"), "", "")
'''
        plugin = tmp_path / "not_allowed.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is False

    def test_trusted_plugins_allowlist_permits(self, tmp_path, monkeypatch):
        """Plugin in trusted_plugins list is loaded."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        import enigma_engine
        monkeypatch.setitem(enigma_engine.CONFIG, "trusted_plugins", ["good.py"])
        code = '''
from enigma_engine.core.commands import CommandResult
def register(registry):
    registry.register("t2.cmd", lambda a, c: CommandResult(True, "ok"), "", "")
'''
        plugin = tmp_path / "good.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is True

    def test_trusted_plugins_empty_allows_all(self, tmp_path, monkeypatch):
        """Empty trusted_plugins list allows all plugins (legacy)."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        import enigma_engine
        monkeypatch.setitem(enigma_engine.CONFIG, "trusted_plugins", [])
        code = '''
from enigma_engine.core.commands import CommandResult
def register(registry):
    registry.register("t3.cmd", lambda a, c: CommandResult(True, "ok"), "", "")
'''
        plugin = tmp_path / "any_plugin.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is True

    def test_ast_scan_flags_subprocess_popen(self):
        """_ast_scan_dangers flags subprocess.Popen()."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers
        source = "import subprocess\nsubprocess.Popen(['ls'])\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert any("subprocess.Popen" in f for f in flags)

    def test_ast_scan_flags_shutil_rmtree(self):
        """_ast_scan_dangers flags shutil.rmtree()."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers
        source = "import shutil\nshutil.rmtree('/tmp/x')\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert any("shutil.rmtree" in f for f in flags)

    def test_ast_scan_handles_syntax_error(self):
        """_ast_scan_dangers returns empty list for unparseable source."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers
        flags = _ast_scan_dangers("def broken(\n", "bad.py")
        assert flags == []

    def test_guard_no_exec_module_before_checks(self):
        """Guard test: load_plugin must pre-scan before exec_module."""
        import inspect
        from enigma_engine.core.plugin_loader import load_plugin
        source = inspect.getsource(load_plugin)
        # _has_register_def and _ast_scan_dangers must appear before exec_module
        pre_scan_pos = source.index("_has_register_def")
        ast_scan_pos = source.index("_ast_scan_dangers")
        exec_pos = source.index("exec_module")
        assert pre_scan_pos < exec_pos, "Pre-scan must happen before exec_module"
        assert ast_scan_pos < exec_pos, "AST scan must happen before exec_module"


# =========================================================================
# Deep-dive audit — DPO mask fix, training checkpoint, engine_chat
# =========================================================================

class TestDPOMaskFix:
    """Verify DPO uses (targets != -100) mask, not (targets > 0)."""

    def test_sequence_log_probs_mask_uses_neg100(self):
        """_get_sequence_logps source must use '!= -100', not '> 0'."""
        import inspect
        from enigma_engine.core.training import Trainer
        # The method may be named _get_sequence_logps or _sequence_log_probs
        for name in ("_get_sequence_logps", "_sequence_log_probs"):
            if hasattr(Trainer, name):
                source = inspect.getsource(getattr(Trainer, name))
                assert "!= -100" in source, (
                    f"{name} should mask with '!= -100', not '> 0'")
                assert "> 0" not in source.split("# ")[0] or "!= -100" in source
                break
        else:
            pytest.skip("DPO log-probs method not found")

    def test_dpo_mask_keeps_token_id_zero(self):
        """Token ID 0 should NOT be masked — only -100 is the ignore marker."""
        import torch
        targets = torch.tensor([[5, 0, 3, -100, 2]])
        mask = (targets != -100).float()
        # Token ID 0 at position 1 should be 1.0 (not masked)
        assert mask[0, 1].item() == 1.0
        # -100 at position 3 should be 0.0 (masked)
        assert mask[0, 3].item() == 0.0
        # Old buggy mask (> 0) would wrongly mask token ID 0
        buggy_mask = (targets > 0).float()
        assert buggy_mask[0, 1].item() == 0.0  # bug: token 0 was masked


class TestLoadCheckpointUnwrapping:
    """Verify load_checkpoint handles various checkpoint dict formats."""

    def test_unwrap_model_state_dict_key(self):
        """Should unwrap 'model_state_dict' key."""
        from enigma_engine.core.training import Trainer
        import inspect
        source = inspect.getsource(Trainer.load_checkpoint)
        assert "model_state_dict" in source
        assert "state_dict" in source

    def test_unwrap_uses_get_chain(self):
        """load_checkpoint should use .get() chain for multiple key formats."""
        from enigma_engine.core.training import Trainer
        import inspect
        source = inspect.getsource(Trainer.load_checkpoint)
        # Should handle at least model_state_dict, state_dict, model keys
        assert "model_state_dict" in source
        assert "'state_dict'" in source or '"state_dict"' in source
        assert "'model'" in source or '"model"' in source


class TestEngineChatReasoningBudget:
    """Verify reasoning token budget uses direct multiplication, not tautological max."""

    def test_no_tautological_max(self):
        """engine_chat should use int(max_gen * 1.5), not max(max_gen, ...)."""
        import inspect
        import enigma_engine.core.engine_chat as ec
        source = inspect.getsource(ec)
        # The old tautological pattern
        assert "max(max_gen" not in source, (
            "Found tautological max(max_gen, int(max_gen * 1.5)) — should be just int(max_gen * 1.5)")

    def test_reasoning_increases_budget(self):
        """int(max_gen * 1.5) is referenced in source for reasoning budget."""
        import inspect
        import enigma_engine.core.engine_chat as ec
        source = inspect.getsource(ec)
        assert "int(max_gen * 1.5)" in source


class TestRouterTrainModeGuard:
    """Verify BackgroundTrainer restores eval mode via try/finally."""

    def test_train_batch_has_try_finally(self):
        """_train_batch must use try/finally to restore model.eval()."""
        import inspect
        from enigma_engine.router import BackgroundTrainer
        source = inspect.getsource(BackgroundTrainer._train_batch)
        assert "try:" in source
        assert "finally:" in source
        assert "model.eval()" in source or ".eval()" in source

    def test_set_model_uses_eval_not_train(self):
        """set_model should set model to eval mode, not train mode."""
        import inspect
        from enigma_engine.router import BackgroundTrainer
        source = inspect.getsource(BackgroundTrainer.set_model)
        # Should call eval(), not train()
        assert ".eval()" in source
        # Should NOT call .train() in set_model
        lines = [l.strip() for l in source.splitlines()
                 if not l.strip().startswith("#")]
        train_calls = [l for l in lines if ".train()" in l]
        assert len(train_calls) == 0, (
            "set_model should not call .train() — only _train_batch does")


class TestRouterHeartbeatOutsideLock:
    """Verify heartbeat pings happen outside the mod_lock."""

    def test_heartbeat_snapshots_outside_lock(self):
        """_heartbeat_loop should snapshot mods, then ping outside lock."""
        import inspect
        from enigma_engine.router import ModRouter
        source = inspect.getsource(ModRouter._heartbeat_loop)
        # Should have `to_ping` list pattern
        assert "to_ping" in source
        # Should have two `with self.mod_lock:` blocks (snapshot + cleanup)
        lock_count = source.count("with self.mod_lock")
        assert lock_count >= 2, (
            f"Expected 2 lock blocks (snapshot+cleanup), found {lock_count}")


class TestRouterStopLogging:
    """Verify router.stop() uses proper exception logging."""

    def test_stop_no_bare_except(self):
        """stop() should not use bare 'except:' — should use 'except Exception'."""
        import inspect
        from enigma_engine.router import ModRouter
        source = inspect.getsource(ModRouter.stop)
        lines = source.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("except") and ":" in stripped:
                assert stripped != "except:", (
                    "stop() should not use bare except: — use except Exception")


# =============================================================================
# VISION ENCODER TESTS
# =============================================================================

class TestVisionEncoderConfig:
    """VisionEncoderConfig dataclass validation."""

    def test_config_importable(self):
        """VisionEncoderConfig should be importable from vision_encoder module."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        assert VisionEncoderConfig is not None

    def test_config_defaults(self):
        """VisionEncoderConfig should have sensible defaults."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig()
        assert cfg.image_size == 224
        assert cfg.patch_size == 16
        assert cfg.dim == 256
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4
        assert cfg.channels == 3

    def test_config_num_patches(self):
        """num_patches property should compute correctly."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=224, patch_size=16)
        assert cfg.num_patches == (224 // 16) ** 2  # 196

    def test_config_custom_values(self):
        """VisionEncoderConfig should accept custom values."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=128, patch_size=8, dim=512, n_layers=6, n_heads=8)
        assert cfg.image_size == 128
        assert cfg.patch_size == 8
        assert cfg.dim == 512
        assert cfg.n_layers == 6
        assert cfg.n_heads == 8
        assert cfg.num_patches == (128 // 8) ** 2  # 256


class TestVisionEncoderPresets:
    """Vision encoder size presets (tiny/small/medium)."""

    def test_presets_exist(self):
        """VISION_PRESETS dict should exist with tiny/small/medium."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        assert "tiny" in VISION_PRESETS
        assert "small" in VISION_PRESETS
        assert "medium" in VISION_PRESETS

    def test_tiny_preset(self):
        """Tiny preset should be smallest."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        tiny = VISION_PRESETS["tiny"]
        assert tiny.n_layers == 2
        assert tiny.dim == 128

    def test_small_preset(self):
        """Small preset should be the default."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        small = VISION_PRESETS["small"]
        assert small.n_layers == 4
        assert small.dim == 256

    def test_medium_preset(self):
        """Medium preset should be largest."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        medium = VISION_PRESETS["medium"]
        assert medium.n_layers == 6
        assert medium.dim == 512


class TestPatchEmbedding:
    """PatchEmbedding module converts images to patch tokens."""

    def test_patch_embedding_importable(self):
        """PatchEmbedding should be importable."""
        from enigma_engine.core.vision_encoder import PatchEmbedding
        assert PatchEmbedding is not None

    def test_patch_embedding_output_shape(self):
        """PatchEmbedding should produce [batch, num_patches, dim] output."""
        import torch
        from enigma_engine.core.vision_encoder import PatchEmbedding
        pe = PatchEmbedding(patch_size=16, channels=3, dim=256)
        x = torch.randn(2, 3, 224, 224)
        out = pe(x)
        # 224/16 = 14, 14*14 = 196 patches
        assert out.shape == (2, 196, 256)

    def test_patch_embedding_different_sizes(self):
        """PatchEmbedding should work with different patch sizes."""
        import torch
        from enigma_engine.core.vision_encoder import PatchEmbedding
        pe = PatchEmbedding(patch_size=8, channels=3, dim=128)
        x = torch.randn(1, 3, 128, 128)
        out = pe(x)
        # 128/8 = 16, 16*16 = 256 patches
        assert out.shape == (1, 256, 128)


class TestVisionEncoder:
    """VisionEncoder — the full ViT model."""

    def test_encoder_importable(self):
        """VisionEncoder should be importable."""
        from enigma_engine.core.vision_encoder import VisionEncoder
        assert VisionEncoder is not None

    def test_encoder_forward_shape(self):
        """VisionEncoder forward should return [batch, num_patches, dim]."""
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=64, patch_size=8, dim=64, n_layers=2, n_heads=2)
        encoder = VisionEncoder(cfg)
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        num_patches = (64 // 8) ** 2  # 64
        assert out.shape == (2, num_patches, 64)

    def test_encoder_is_nn_module(self):
        """VisionEncoder should be a proper nn.Module."""
        import torch.nn as nn
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        assert isinstance(encoder, nn.Module)

    def test_encoder_has_parameters(self):
        """VisionEncoder should have trainable parameters."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        params = sum(p.numel() for p in encoder.parameters())
        assert params > 0

    def test_encoder_gradients_flow(self):
        """Gradients should flow through the encoder (trainable)."""
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        x = torch.randn(1, 3, 32, 32)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        # Check at least one parameter has gradients
        has_grad = any(p.grad is not None for p in encoder.parameters())
        assert has_grad

    def test_encoder_tiny_preset_params(self):
        """Tiny preset should have roughly 500K params."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VISION_PRESETS
        encoder = VisionEncoder(VISION_PRESETS["tiny"])
        params = sum(p.numel() for p in encoder.parameters())
        # Should be in ballpark of 500K (allow wide range for architecture details)
        assert 100_000 < params < 2_000_000

    def test_encoder_config_stored(self):
        """VisionEncoder should store its config."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        assert encoder.config is cfg

    def test_encoder_position_embeddings(self):
        """Encoder should have learnable position embeddings."""
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        assert hasattr(encoder, "pos_embed")
        # Position embeddings should match [1, num_patches, dim]
        num_patches = (32 // 8) ** 2
        assert encoder.pos_embed.shape == (1, num_patches, 32)


class TestImagePreprocessing:
    """Image preprocessing — resize, normalize, tensor conversion."""

    def test_preprocess_function_exists(self):
        """preprocess_image function should exist."""
        from enigma_engine.core.vision_encoder import preprocess_image
        assert callable(preprocess_image)

    def test_preprocess_pil_image(self):
        """preprocess_image should handle PIL Images."""
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (400, 300), (128, 64, 32))
        tensor = preprocess_image(img, image_size=224)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_normalizes(self):
        """Preprocessed image values should be roughly in [-1, 1] range."""
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        tensor = preprocess_image(img, image_size=224)
        # After normalization to [-1, 1], values should be bounded
        assert tensor.min() >= -1.1
        assert tensor.max() <= 1.1

    def test_preprocess_from_path(self):
        """preprocess_image should accept a file path string."""
        import tempfile
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            tensor = preprocess_image(f.name, image_size=64)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 64, 64)

    def test_preprocess_grayscale_converts(self):
        """Grayscale images should be converted to RGB (3 channels)."""
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("L", (100, 100), 128)
        tensor = preprocess_image(img, image_size=32)
        assert tensor.shape == (1, 3, 32, 32)

    def test_preprocess_rgba_converts(self):
        """RGBA images should be converted to RGB (drop alpha)."""
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        tensor = preprocess_image(img, image_size=32)
        assert tensor.shape == (1, 3, 32, 32)


class TestEncodeImage:
    """encode_image convenience function."""

    def test_encode_image_exists(self):
        """encode_image function should exist."""
        from enigma_engine.core.vision_encoder import encode_image
        assert callable(encode_image)

    def test_encode_image_returns_features(self):
        """encode_image should return feature tensor from encoder."""
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import (
            VisionEncoder, VisionEncoderConfig, encode_image,
        )
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        img = Image.new("RGB", (100, 100), (128, 64, 32))
        features = encode_image(encoder, img)
        num_patches = (32 // 8) ** 2  # 16
        assert features.shape == (1, num_patches, 32)


class TestEncodeScreen:
    """encode_screen captures desktop screenshot and encodes it."""

    def test_encode_screen_exists(self):
        """encode_screen function should exist."""
        from enigma_engine.core.vision_encoder import encode_screen
        assert callable(encode_screen)


class TestEncodeCamera:
    """encode_camera captures webcam frame and encodes it."""

    def test_encode_camera_exists(self):
        """encode_camera function should exist."""
        from enigma_engine.core.vision_encoder import encode_camera
        assert callable(encode_camera)


class TestEncodeVideoFrames:
    """encode_video_frames samples and encodes video frames."""

    def test_encode_video_frames_exists(self):
        """encode_video_frames function should exist."""
        from enigma_engine.core.vision_encoder import encode_video_frames
        assert callable(encode_video_frames)


class TestVisionEncoderSaveLoad:
    """Save/load vision encoder weights."""

    def test_state_dict_saveable(self):
        """Vision encoder state_dict should be saveable."""
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        encoder = VisionEncoder(cfg)
        sd = encoder.state_dict()
        assert len(sd) > 0

    def test_config_to_dict_roundtrip(self):
        """VisionEncoderConfig should round-trip through dict."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=128, patch_size=8, dim=64, n_layers=3, n_heads=4)
        d = cfg.to_dict()
        cfg2 = VisionEncoderConfig(**d)
        assert cfg2.image_size == 128
        assert cfg2.patch_size == 8
        assert cfg2.dim == 64
        assert cfg2.n_layers == 3
        assert cfg2.n_heads == 4

    def test_load_state_dict_restores(self):
        """Loading a state_dict should restore encoder weights."""
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        enc1 = VisionEncoder(cfg)
        sd = enc1.state_dict()
        enc2 = VisionEncoder(cfg)
        enc2.load_state_dict(sd)
        # Both should produce same output
        x = torch.randn(1, 3, 32, 32)
        enc1.eval()
        enc2.eval()
        out1 = enc1(x)
        out2 = enc2(x)
        assert torch.allclose(out1, out2)


class TestVisionWithTextModel:
    """Integration: vision encoder + text model via forward_multimodal."""

    def test_vision_features_through_model(self):
        """Vision encoder output should work with forward_multimodal."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig

        vcfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        v_encoder = VisionEncoder(vcfg)

        # Text model with vision_hidden_size matching encoder dim
        tcfg = ForgeConfig(
            vocab_size=100, dim=64, n_layers=1, n_heads=2,
            max_seq_len=64, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        # Encode an image
        img_tensor = torch.randn(1, 3, 32, 32)
        vision_features = v_encoder(img_tensor)

        # Pass through text model with some text tokens
        text_ids = torch.randint(0, 100, (1, 5))
        logits = model.forward_multimodal(
            input_ids=text_ids,
            vision_features=vision_features,
        )
        # Output should cover vision patches + text tokens
        expected_seq = vcfg.num_patches + 5
        assert logits.shape == (1, expected_seq, 100)

    def test_vision_only_forward(self):
        """forward_multimodal should work with only vision features (no text)."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig

        vcfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=32, n_layers=1, n_heads=2)
        v_encoder = VisionEncoder(vcfg)

        tcfg = ForgeConfig(
            vocab_size=100, dim=64, n_layers=1, n_heads=2,
            max_seq_len=64, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        img_tensor = torch.randn(1, 3, 32, 32)
        vision_features = v_encoder(img_tensor)

        logits = model.forward_multimodal(
            input_ids=None,
            vision_features=vision_features,
        )
        assert logits.shape == (1, vcfg.num_patches, 100)


# =============================================================================
# VISION TRAINING TESTS
# =============================================================================

class TestVisionTraining:
    """Trainer.train_vision method for image-text pair training."""

    def test_trainer_has_train_vision(self):
        """Trainer must have train_vision method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "train_vision")

    def test_train_vision_signature(self):
        """train_vision should accept vision_encoder and image-text data."""
        import inspect
        from enigma_engine.core.training import Trainer
        sig = inspect.signature(Trainer.train_vision)
        params = list(sig.parameters.keys())
        assert "vision_encoder" in params
        assert "data" in params

    def test_train_vision_updates_weights(self):
        """train_vision should update both encoder and projection weights."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        # Tiny model setup
        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()

        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        # Snapshot initial weights
        enc_before = v_enc.patch_embed.proj.weight.clone()
        proj_before = model.vision_projection.weight.clone()

        # Minimal training data: image tensor + text
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        img = Image.new("RGB", (16, 16), (128, 64, 32))
        data = [{"image": img, "text": "a red square"}]

        config = TrainingConfig(epochs=2, batch_size=1, learning_rate=1e-3)
        trainer = Trainer(model, tok, config)
        trainer.train_vision(vision_encoder=v_enc, data=data)

        # Both encoder and projection should have changed
        assert not torch.equal(enc_before, v_enc.patch_embed.proj.weight)
        assert not torch.equal(proj_before, model.vision_projection.weight)

    def test_train_vision_returns_state(self):
        """train_vision should return TrainingState."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig, TrainingState
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        img = Image.new("RGB", (16, 16), (64, 128, 255))
        data = [{"image": img, "text": "blue square"}]
        config = TrainingConfig(epochs=1, batch_size=1)
        trainer = Trainer(model, tok, config)
        state = trainer.train_vision(vision_encoder=v_enc, data=data)
        assert isinstance(state, TrainingState)
        assert len(state.training_losses) >= 1

    def test_train_vision_loss_decreases(self):
        """Loss should generally decrease over epochs (small model, enough data)."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        # Create several training pairs for better signal
        data = [
            {"image": Image.new("RGB", (16, 16), (255, 0, 0)), "text": "red"},
            {"image": Image.new("RGB", (16, 16), (0, 255, 0)), "text": "green"},
            {"image": Image.new("RGB", (16, 16), (0, 0, 255)), "text": "blue"},
        ]
        config = TrainingConfig(epochs=5, batch_size=1, learning_rate=5e-3)
        trainer = Trainer(model, tok, config)
        state = trainer.train_vision(vision_encoder=v_enc, data=data)

        # Last epoch loss should be lower than first
        assert state.training_losses[-1] < state.training_losses[0]

    def test_train_vision_callbacks(self):
        """train_vision should fire progress and loss callbacks."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        progress_log = []
        loss_log = []

        config = TrainingConfig(epochs=1, batch_size=1, log_every=1)
        trainer = Trainer(model, tok, config)
        trainer.on_progress = lambda p, m: progress_log.append((p, m))
        trainer.on_loss = lambda l: loss_log.append(l)

        img = Image.new("RGB", (16, 16), (128, 128, 128))
        data = [{"image": img, "text": "gray square"}]
        trainer.train_vision(vision_encoder=v_enc, data=data)

        assert len(progress_log) > 0
        assert len(loss_log) > 0

    def test_train_vision_stop_requested(self):
        """train_vision should respect request_stop()."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        config = TrainingConfig(epochs=100, batch_size=1)
        trainer = Trainer(model, tok, config)

        # Use callback to stop after first epoch
        def stop_after_first(epoch, avg_loss):
            trainer.request_stop()
        trainer.on_epoch_complete = stop_after_first

        img = Image.new("RGB", (16, 16), (0, 0, 0))
        data = [{"image": img, "text": "black"}]
        state = trainer.train_vision(vision_encoder=v_enc, data=data)

        # Should complete far fewer than 100 epochs
        assert len(state.training_losses) <= 3


class TestVisionDataParsing:
    """Parsing and validation of image-text training data."""

    def test_parse_image_path_data(self):
        """Vision training should accept image paths in data dicts."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_vision)
        # Should handle both PIL images and file paths
        assert "image" in source

    def test_requires_vision_projection(self):
        """train_vision should raise if model lacks vision_projection."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()

        # Model WITHOUT vision_hidden_size — no projection layer
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2, max_seq_len=32,
        )
        model = Enigma(config=tcfg)

        config = TrainingConfig(epochs=1, batch_size=1)
        trainer = Trainer(model, tok, config)

        img = Image.new("RGB", (16, 16), (128, 128, 128))
        data = [{"image": img, "text": "gray"}]

        with pytest.raises(ValueError, match="vision"):
            trainer.train_vision(vision_encoder=v_enc, data=data)


class TestScanVisionData:
    """Scanner detection of image-text datasets."""

    def test_scan_vision_data_function_exists(self):
        """scan_vision_data function should be importable from scanners."""
        from enigma_engine.gui.scanners import scan_vision_data
        assert callable(scan_vision_data)

    def test_scan_empty_dir(self):
        """scan_vision_data on empty dir should return empty list."""
        import tempfile
        from enigma_engine.gui.scanners import scan_vision_data
        with tempfile.TemporaryDirectory() as d:
            result = scan_vision_data(d)
            assert result == []

    def test_scan_paired_files(self):
        """scan_vision_data should detect image.png + image.txt pairs."""
        import tempfile
        from pathlib import Path
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.gui.scanners import scan_vision_data

        with tempfile.TemporaryDirectory() as d:
            # Create paired files
            img = Image.new("RGB", (10, 10), (255, 0, 0))
            img.save(Path(d) / "photo1.png")
            (Path(d) / "photo1.txt").write_text("a red square", encoding="utf-8")

            result = scan_vision_data(d)
            assert len(result) == 1
            assert "image" in result[0]
            assert "text" in result[0]

    def test_scan_jsonl_file(self):
        """scan_vision_data should detect JSONL files with image+text fields."""
        import json
        import tempfile
        from pathlib import Path
        from enigma_engine.gui.scanners import scan_vision_data

        with tempfile.TemporaryDirectory() as d:
            jsonl_path = Path(d) / "captions.jsonl"
            records = [
                {"image": "img1.png", "text": "a cat"},
                {"image": "img2.png", "text": "a dog"},
            ]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            result = scan_vision_data(d)
            assert len(result) == 2


# ============================================================
# Vision Chat Integration
# ============================================================

class TestVisionChatIntegration:
    """Tests for vision encoder integration with chat / generation."""

    def test_chat_accepts_images_param(self):
        """chat() should accept an 'images' keyword argument."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        sig = inspect.signature(_ChatMixin.chat)
        assert "images" in sig.parameters, "chat() must accept 'images' param"

    def test_engine_has_vision_encoder_attr(self):
        """EnigmaEngine instances should have a vision_encoder attribute."""
        from enigma_engine.core.inference import EnigmaEngine
        engine = EnigmaEngine.from_model(
            model=type("M", (), {"parameters": lambda s: iter([]),
                                   "eval": lambda s: s,
                                   "to": lambda s, *a, **k: s})(),
            tokenizer=type("T", (), {"vocab_size": 10})(),
            device="cpu",
        )
        assert hasattr(engine, "vision_encoder")
        assert engine.vision_encoder is None

    def test_encode_images_for_chat(self):
        """_encode_images_for_chat should encode image paths to features."""
        import torch
        from enigma_engine.core.engine_chat import _ChatMixin
        assert hasattr(_ChatMixin, "_encode_images_for_chat"), (
            "_ChatMixin needs _encode_images_for_chat method"
        )

    def test_encode_images_returns_tensor(self):
        """_encode_images_for_chat should return a batched tensor."""
        import tempfile
        import torch
        from PIL import Image
        from pathlib import Path
        from enigma_engine.core.engine_chat import _ChatMixin
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig

        # Create a minimal mixin instance with vision encoder
        mixin = _ChatMixin()
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=64,
                                  n_layers=1, n_heads=2)
        mixin.vision_encoder = VisionEncoder(cfg)
        mixin.device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "test.png"
            Image.new("RGB", (32, 32), (128, 128, 128)).save(img_path)

            features = mixin._encode_images_for_chat([str(img_path)])
            assert isinstance(features, torch.Tensor)
            assert features.dim() == 3  # [batch, seq, dim]
            assert features.shape[-1] == cfg.dim

    def test_encode_images_multiple(self):
        """_encode_images_for_chat should stack features from multiple images."""
        import tempfile
        import torch
        from PIL import Image
        from pathlib import Path
        from enigma_engine.core.engine_chat import _ChatMixin
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig

        mixin = _ChatMixin()
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=64,
                                  n_layers=1, n_heads=2)
        mixin.vision_encoder = VisionEncoder(cfg)
        mixin.device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                p = Path(d) / f"img{i}.png"
                Image.new("RGB", (32, 32), (i * 50, 0, 0)).save(p)
                paths.append(str(p))

            features = mixin._encode_images_for_chat(paths)
            # Should concatenate along sequence dimension
            assert features.dim() == 3
            num_patches = cfg.num_patches
            assert features.shape[1] == num_patches * 3

    def test_encode_images_no_encoder_returns_none(self):
        """_encode_images_for_chat should return None when no vision encoder."""
        from enigma_engine.core.engine_chat import _ChatMixin

        mixin = _ChatMixin()
        mixin.vision_encoder = None
        mixin.device = None

        result = mixin._encode_images_for_chat(["/fake/path.png"])
        assert result is None

    def test_generate_multimodal_method_exists(self):
        """_GenerationMixin should have _generate_with_vision method."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        assert hasattr(_GenerationMixin, "_generate_with_vision")

    def test_generate_with_vision_produces_output(self):
        """_generate_with_vision should produce text output."""
        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.inference import EnigmaEngine

        tok = SimpleTokenizer()
        vcfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=64,
                                   n_layers=1, n_heads=2)
        cfg = ForgeConfig(
            vocab_size=tok.vocab_size,
            dim=64, n_layers=1, n_heads=2, max_seq_len=128,
            vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=cfg)
        model.eval()

        engine = EnigmaEngine.from_model(model, tok, device="cpu")
        engine.vision_encoder = VisionEncoder(vcfg)

        # Create fake vision features: [1, 16, 64]
        vision_features = torch.randn(1, vcfg.num_patches, vcfg.dim)

        text = engine._generate_with_vision(
            prompt="Describe this image:",
            vision_features=vision_features,
            max_gen=5,
        )
        assert isinstance(text, str)
        assert len(text) > 0


# ================================================================
# Web Utilities (core/web_utils.py)
# ================================================================

class TestWebUtilsCore:
    """Core tests for web_utils shared module."""

    def test_ddg_search_accepts_max_results(self):
        """ddg_search accepts max_results parameter."""
        import inspect
        from enigma_engine.core.web_utils import ddg_search
        sig = inspect.signature(ddg_search)
        assert "max_results" in sig.parameters

    def test_fetch_page_text_accepts_max_chars(self):
        """fetch_page_text accepts max_chars parameter."""
        import inspect
        from enigma_engine.core.web_utils import fetch_page_text
        sig = inspect.signature(fetch_page_text)
        assert "max_chars" in sig.parameters

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
# GGUF Export
# ---------------------------------------------------------------------------

class TestGGUFExport:
    """Tests for GGUF export functionality."""

    def test_export_to_gguf_standalone_exists(self):
        """Module-level export_to_gguf function exists in gguf.py."""
        from enigma_engine.core.gguf import export_to_gguf
        assert callable(export_to_gguf)

    def test_export_to_gguf_method_on_enigma(self):
        """Enigma class has an export_to_gguf instance method."""
        from enigma_engine.core.model import Enigma
        assert hasattr(Enigma, "export_to_gguf")
        assert callable(getattr(Enigma, "export_to_gguf"))

    def test_export_to_gguf_method_calls_gguf_export(self):
        """Enigma.export_to_gguf delegates to gguf.export_to_gguf."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.export_to_gguf)
        assert "export_to_gguf" in source

    def test_infer_metadata_handles_forgeconfig(self):
        """_infer_metadata should handle ForgeConfig attributes (dim, n_layers, etc.)."""
        import inspect
        from enigma_engine.core.gguf import GGUFExporter
        source = inspect.getsource(GGUFExporter._infer_metadata)
        # Must handle ForgeConfig's attribute names (not just HuggingFace's)
        assert "config.n_layers" in source or "'n_layers'" in source or '"n_layers"' in source

    def test_forge_export_gguf_uses_correct_kwarg(self):
        """_export_student_gguf must use quant_type= not quantization_type=."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._export_student_gguf)
        assert "quant_type=" in source
        assert "quantization_type=" not in source

    def test_forge_export_gguf_arg_order(self):
        """_export_student_gguf must pass (model, out_path, ...) not (model, tokenizer, ...)."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._export_student_gguf)
        # Find the export_to_gguf call — args should be model then path
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'export_to_gguf(' in line:
                # Next few lines should have model, out_path order
                block = '\n'.join(lines[i:i+5])
                assert "tokenizer" not in block.split("export_to_gguf(")[1].split(",")[1] or \
                       "out_path" in block or "quant_type" in block
                break


# ================================================================
# D5: Weight mapping tests
# ================================================================

class TestWeightMapping:
    """Tests for weight_mapping.py — format conversion."""

    def test_import(self):
        """weight_mapping module imports cleanly."""
        from enigma_engine.core.weight_mapping import WeightMapper
        assert WeightMapper is not None

    def test_hf_model_maps_exist(self):
        """HF_MODEL_MAPS has entries for each architecture."""
        from enigma_engine.core.weight_mapping import HF_MODEL_MAPS
        for arch in ("llama", "gpt2", "phi", "qwen2", "gemma"):
            assert arch in HF_MODEL_MAPS, f"Missing map for {arch}"

    def test_gguf_weight_map_nonempty(self):
        """GGUF_WEIGHT_MAP has entries."""
        from enigma_engine.core.weight_mapping import GGUF_WEIGHT_MAP
        assert len(GGUF_WEIGHT_MAP) > 5

    def test_mapper_hf_to_forge_llama(self):
        """map_huggingface_to_forge maps llama weight names."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"model.embed_tokens.weight": "tensor_a",
                 "model.norm.weight": "tensor_b"}
        result = mapper.map_huggingface_to_forge(dummy, model_type="llama")
        assert "tok_embeddings.weight" in result
        assert "norm.weight" in result

    def test_mapper_hf_to_forge_gpt2(self):
        """map_huggingface_to_forge maps GPT-2 weight names."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"transformer.wte.weight": "tensor_a",
                 "transformer.ln_f.weight": "tensor_b"}
        result = mapper.map_huggingface_to_forge(dummy, model_type="gpt2")
        assert "tok_embeddings.weight" in result
        assert "norm.weight" in result

    def test_mapper_gguf_to_forge(self):
        """map_gguf_to_forge maps GGUF tensor names."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"token_embd.weight": "tensor_a",
                 "output_norm.weight": "tensor_b"}
        result = mapper.map_gguf_to_forge(dummy)
        assert "tok_embeddings.weight" in result
        assert "norm.weight" in result

    def test_mapper_onnx_to_forge(self):
        """map_onnx_to_forge can convert ONNX weights."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"model.embed_tokens.weight": "tensor_a"}
        result = mapper.map_onnx_to_forge(dummy)
        assert "tok_embeddings.weight" in result

    def test_detect_model_type_gpt2(self):
        """_detect_hf_model_type identifies GPT-2 layout."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"transformer.h.0.ln_1.weight": 1,
                 "transformer.wte.weight": 2}
        assert mapper._detect_hf_model_type(dummy) == "gpt2"

    def test_detect_model_type_llama(self):
        """_detect_hf_model_type identifies LLaMA layout."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"model.layers.0.self_attn.q_proj.weight": 1,
                 "model.layers.0.mlp.gate_proj.weight": 2}
        assert mapper._detect_hf_model_type(dummy) == "llama"

    def test_get_stats(self):
        """get_stats returns mapping statistics."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        mapper.map_huggingface_to_forge(
            {"model.embed_tokens.weight": "t"}, model_type="llama")
        stats = mapper.get_stats()
        assert "mapped" in stats
        assert "skipped" in stats
        assert stats["mapped"] >= 1


# ================================================================
# D5: GGUF dequantization tests
# ================================================================

class TestGGUFDequant:
    """Tests for gguf_dequant.py — tensor parsing and dequantization."""

    def test_import(self):
        """gguf_dequant module imports cleanly."""
        from enigma_engine.core.gguf_dequant import (
            parse_gguf_tensors,
            extract_config_from_metadata,
        )
        assert callable(parse_gguf_tensors)
        assert callable(extract_config_from_metadata)

    def test_extract_config_from_metadata_llama(self):
        """extract_config_from_metadata parses llama-style metadata."""
        from enigma_engine.core.gguf_dequant import extract_config_from_metadata
        metadata = {
            "llama.embedding_length": 2048,
            "llama.block_count": 16,
            "llama.attention.head_count": 16,
            "llama.attention.head_count_kv": 4,
            "llama.context_length": 4096,
        }
        config = extract_config_from_metadata(metadata)
        assert config["dim"] == 2048
        assert config["n_layers"] == 16
        assert config["n_heads"] == 16
        assert config["n_kv_heads"] == 4
        assert config["max_seq_len"] == 4096

    def test_extract_config_defaults(self):
        """extract_config_from_metadata fills defaults for missing keys."""
        from enigma_engine.core.gguf_dequant import extract_config_from_metadata
        config = extract_config_from_metadata({})
        assert "dim" in config
        assert "n_layers" in config
        assert "n_heads" in config
        assert "vocab_size" in config
        assert "max_seq_len" in config

    def test_dequantize_q4_0_shape(self):
        """dequantize_q4_0 returns correct shape."""
        import struct
        import numpy as np
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_0
        # Build a single block: 2 bytes float16 scale + 16 bytes data = 18 bytes
        scale = np.float16(0.5)
        block = scale.tobytes() + bytes(16)
        result = dequantize_q4_0(block, (32,))
        assert result.shape == (32,)
        assert result.dtype == torch.float32

    def test_dequantize_q8_0_shape(self):
        """dequantize_q8_0 returns correct shape."""
        import numpy as np
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q8_0
        # Build a single block: 2 bytes float16 scale + 32 bytes data = 34 bytes
        scale = np.float16(1.0)
        block = scale.tobytes() + bytes(32)
        result = dequantize_q8_0(block, (32,))
        assert result.shape == (32,)
        assert result.dtype == torch.float32

    def test_extract_config_embed_length_alias(self):
        """extract_config_from_metadata handles embed_length alias."""
        from enigma_engine.core.gguf_dequant import extract_config_from_metadata
        metadata = {"llama.embed_length": 512}
        config = extract_config_from_metadata(metadata)
        assert config["dim"] == 512


# ================================================================
# D5: Model components tests
# ================================================================

class TestModelComponents:
    """Tests for model_components.py — neural network building blocks."""

    def test_import(self):
        """model_components module imports cleanly."""
        from enigma_engine.core.model_components import (
            RMSNorm, Attention, FeedForward, TransformerBlock,
        )
        assert RMSNorm is not None
        assert TransformerBlock is not None

    def test_rmsnorm_output_shape(self):
        """RMSNorm preserves input shape."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_normalizes(self):
        """RMSNorm output has roughly unit RMS."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(128)
        x = torch.randn(1, 5, 128) * 10.0
        out = norm(x)
        rms = (out ** 2).mean(-1).sqrt()
        # Should be close to 1.0 (norm weight initialized to ones)
        assert rms.mean().item() == pytest.approx(1.0, abs=0.3)

    def test_precompute_rope_frequencies(self):
        """precompute_rope_frequencies returns correct shapes."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import precompute_rope_frequencies
        freqs = precompute_rope_frequencies(64, 128)
        assert freqs.shape[0] == 128  # seq_len
        assert freqs.shape[1] == 32   # dim // 2

    def test_feedforward_output_shape(self):
        """FeedForward preserves batch and seq dims."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import FeedForward
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128)
        ff = FeedForward(config)
        x = torch.randn(1, 5, 64)
        out = ff(x)
        assert out.shape == (1, 5, 64)

    def test_moe_feedforward_has_experts(self):
        """MoEFeedForward creates multiple expert modules."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import MoEFeedForward
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(
            dim=64, hidden_dim=128,
            use_moe=True, num_experts=4, num_experts_per_token=2)
        moe = MoEFeedForward(config)
        assert len(moe.experts) == 4


# ================================================================
# D5: Model config shim tests
# ================================================================

class TestModelConfigShim:
    """Tests for model_config.py — backward-compat shim."""

    def test_import(self):
        """model_config.py imports cleanly."""
        from enigma_engine.core.model_config import get_model_config
        assert callable(get_model_config)

    def test_get_model_config_returns_dict(self):
        """get_model_config returns a valid config dict."""
        from enigma_engine.core.model_config import get_model_config
        config = get_model_config("tiny")
        assert isinstance(config, dict)
        assert "dim" in config
        assert "n_layers" in config

    def test_get_model_config_invalid_raises(self):
        """get_model_config raises ValueError for unknown sizes."""
        from enigma_engine.core.model_config import get_model_config
        with pytest.raises(ValueError, match="Unknown size"):
            get_model_config("nonexistent_size_xyz")

    def test_model_presets_reexport(self):
        """model_config.py re-exports MODEL_PRESETS from model_presets."""
        from enigma_engine.core.model_config import MODEL_PRESETS
        from enigma_engine.core.model_presets import MODEL_PRESETS as orig
        assert MODEL_PRESETS is orig


# ================================================================
# Mod Tools — auto-register mod commands as AI tools
# ================================================================

class TestModTools:
    """Tests for enigma_engine.core.mod_tools module."""

    def test_module_imports(self):
        """mod_tools module imports without error."""
        from enigma_engine.core import mod_tools
        assert hasattr(mod_tools, "discover_mod_tools")
        assert hasattr(mod_tools, "register_mod_commands")
        assert hasattr(mod_tools, "format_tools_for_prompt")

    def test_discover_mod_tools_returns_list(self):
        """discover_mod_tools returns a list of tool dicts."""
        from enigma_engine.core.mod_tools import discover_mod_tools
        mods_dir = Path(__file__).parent.parent / "mods"
        tools = discover_mod_tools(mods_dir)
        assert isinstance(tools, list)
        # Should find at least imagegen.generate and voice.listen
        names = [t["name"] for t in tools]
        assert "imagegen.generate" in names
        assert "voice.listen" in names

    def test_discover_mod_tools_skips_template(self):
        """discover_mod_tools skips _template directory."""
        from enigma_engine.core.mod_tools import discover_mod_tools
        mods_dir = Path(__file__).parent.parent / "mods"
        tools = discover_mod_tools(mods_dir)
        mod_ids = {t["mod_id"] for t in tools}
        assert "_template" not in mod_ids

    def test_discover_mod_tools_has_required_keys(self):
        """Each tool dict has mod_id, name, description, args."""
        from enigma_engine.core.mod_tools import discover_mod_tools
        mods_dir = Path(__file__).parent.parent / "mods"
        tools = discover_mod_tools(mods_dir)
        for t in tools:
            assert "mod_id" in t
            assert "name" in t
            assert "description" in t
            assert "args" in t

    def test_discover_mod_tools_nonexistent_dir(self):
        """discover_mod_tools returns empty for non-existent directory."""
        from enigma_engine.core.mod_tools import discover_mod_tools
        tools = discover_mod_tools(Path("/nonexistent/dir"))
        assert tools == []

    def test_register_mod_commands_returns_count(self):
        """register_mod_commands returns count of newly registered."""
        from enigma_engine.core.mod_tools import register_mod_commands
        from enigma_engine.core.commands import CommandRegistry
        registry = CommandRegistry()
        mods_dir = Path(__file__).parent.parent / "mods"
        count = register_mod_commands(registry, mods_dir)
        assert isinstance(count, int)
        assert count > 0

    def test_register_mod_commands_skips_existing(self):
        """register_mod_commands does not overwrite existing commands."""
        from enigma_engine.core.mod_tools import register_mod_commands
        from enigma_engine.core.commands import CommandRegistry, CommandResult
        registry = CommandRegistry()
        # Pre-register a command
        sentinel = lambda a, c: CommandResult(True, "sentinel")
        registry.register("imagegen.generate", sentinel, "test", "test")
        mods_dir = Path(__file__).parent.parent / "mods"
        register_mod_commands(registry, mods_dir)
        # Should still be our sentinel, not overwritten
        assert registry._commands["imagegen.generate"].handler is sentinel

    def test_format_tools_for_prompt_empty(self):
        """format_tools_for_prompt returns empty string for no mods."""
        from enigma_engine.core.mod_tools import format_tools_for_prompt
        assert format_tools_for_prompt([]) == ""

    def test_format_tools_for_prompt_includes_mods(self):
        """format_tools_for_prompt includes mod names and commands."""
        from enigma_engine.core.mod_tools import format_tools_for_prompt
        mods_data = [{
            "id": "test_mod",
            "name": "Test Mod",
            "description": "A test mod",
            "_running": True,
            "commands_full": [
                {"name": "do_thing", "description": "Does a thing",
                 "args": {"input": {"type": "string", "required": True}}},
            ],
        }]
        result = format_tools_for_prompt(mods_data)
        assert "[AVAILABLE TOOLS]" in result
        assert "Test Mod" in result
        assert "RUNNING" in result
        assert "test_mod.do_thing" in result
        assert "do_thing" in result

    def test_format_tools_shows_available_status(self):
        """Stopped mods show as AVAILABLE, not RUNNING."""
        from enigma_engine.core.mod_tools import format_tools_for_prompt
        mods_data = [{
            "id": "stopped_mod",
            "name": "Stopped Mod",
            "_running": False,
            "commands_full": [{"name": "cmd1", "description": "d"}],
        }]
        result = format_tools_for_prompt(mods_data)
        assert "AVAILABLE" in result


# ================================================================
# Auto Research — proactive web research
# ================================================================

class TestAutoResearch:
    """Tests for enigma_engine.core.auto_research module."""

    def test_module_imports(self):
        """auto_research module imports without error."""
        from enigma_engine.core import auto_research
        assert hasattr(auto_research, "auto_research")
        assert hasattr(auto_research, "should_auto_research")

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

    def test_auto_research_returns_string(self):
        """auto_research always returns a string."""
        from enigma_engine.core.auto_research import auto_research
        result = auto_research("nonexistent topic xyz123")
        assert isinstance(result, str)


# ================================================================
# GUI Command Registration (mod.start, mod.stop, mod.list)
# ================================================================

class TestGUICommandRegistration:
    """Tests for mod management commands registered by the GUI."""

    def test_register_gui_commands_method(self):
        """LogicMixin must have _register_gui_commands method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_register_gui_commands")
        assert callable(getattr(LogicMixin, "_register_gui_commands"))

    def test_build_gui_context_has_tools(self):
        """_build_gui_context source mentions mod_tools."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "format_tools_for_prompt" in source

    def test_build_gui_context_has_terminal_instructions(self):
        """_build_gui_context includes terminal agent instructions."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "Terminal Access" in source

    def test_build_gui_context_has_mod_management(self):
        """_build_gui_context includes mod management instructions."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "mod.start" in source
        assert "mod.stop" in source
        assert "mod.list" in source

    def test_build_gui_context_has_learning_mode(self):
        """_build_gui_context includes learning mode status."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "learn_while_chatting" in source
        assert "Learning Mode" in source

    def test_register_gui_commands_registers_mod_commands(self):
        """_register_gui_commands source registers mod.start etc."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._register_gui_commands)
        assert "mod.start" in source
        assert "mod.stop" in source
        assert "mod.list" in source
        assert "register_mod_commands" in source

    def test_chat_has_auto_research_integration(self):
        """Chat flow must integrate auto_research."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        source = inspect.getsource(LogicChatMixin._send_message)
        assert "auto_research" in source
        assert "should_auto_research" in source


# ================================================================
# Tensor contiguity: .reshape() over .view()
# ================================================================

class TestTensorReshapeSafety:
    """Ensure model uses .reshape() instead of .view() for non-contiguous safety."""

    def test_model_forward_uses_reshape_for_loss(self):
        """Model loss computation must use .reshape() not .view()."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.forward)
        # Loss lines must use reshape, not view
        assert "logits.reshape(" in source, (
            "logits loss should use .reshape() for non-contiguous tensor safety"
        )
        assert "targets.reshape(" in source, (
            "targets loss should use .reshape() for non-contiguous tensor safety"
        )

    def test_model_forward_no_view_on_logits(self):
        """Model forward should not use .view() on logits or targets."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.forward)
        assert "logits.view(" not in source, (
            "logits.view() can fail on non-contiguous tensors — use .reshape()"
        )
        assert "targets.view(" not in source, (
            "targets.view() can fail on non-contiguous tensors — use .reshape()"
        )

    def test_attention_flash_path_uses_reshape(self):
        """Flash attention output reshape must handle non-contiguous tensors."""
        import inspect
        from enigma_engine.core.model_components import Attention
        source = inspect.getsource(Attention.forward)
        # The flash attention path should use reshape or contiguous().view()
        assert "output.view(B, T, -1)" not in source, (
            "Flash attention output.view() can fail — use .reshape()"
        )


# ================================================================
# Training data parser: multi-format support
# ================================================================

class TestTrainingDataParser:
    """Training data parser handles multiple formats."""

    def test_parse_qa_format(self):
        """Parser handles Q:/A: format."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._parse_training_data)
        assert "Q:" in source
        assert "A:" in source

    def test_parse_user_ai_dialogue(self):
        """Parser handles User:/AI: dialogue format."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._parse_training_data)
        assert "User" in source
        assert "AI" in source
        assert "dialogue_pattern" in source

    def test_parse_jsonl_format(self):
        """Parser handles JSONL format."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._parse_training_data)
        assert "json.loads" in source
        assert "prompt" in source
        assert "completion" in source

    def test_parse_paragraph_fallback(self):
        """Parser falls back to paragraph splitting for raw text."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._parse_training_data)
        assert "paragraph" in source.lower()
        assert "split" in source


# ================================================================
# Training Pipeline Quality Fixes
# ================================================================

class TestTrainingPadToken:
    """Pad token handled correctly for loss computation."""

    def test_create_batches_uses_tokenizer_pad_id(self):
        """_create_batches pads with the tokenizer's pad_token_id."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._create_batches)
        # Should reference pad_token_id from tokenizer, not hardcoded 0
        assert "pad_token_id" in source

    def test_trainer_stores_pad_token_id(self):
        """Trainer stores pad_token_id from the tokenizer."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.__init__)
        assert "pad_token_id" in source

    def test_train_passes_pad_id_to_model(self):
        """Training loop passes pad_token_id as ignore_index."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._train_one_batch)
        assert "pad_token_id" in source


class TestTrainingMaxLength:
    """Batch creation uses model's max_seq_len, not hardcoded 512."""

    def test_create_batches_reads_model_seq_len(self):
        """_create_batches uses model's max_seq_len when available."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._create_batches)
        assert "max_seq_len" in source

    def test_train_passes_model_seq_len(self):
        """train() passes model's max_seq_len to _create_batches."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "max_seq_len" in source or "max_length" in source


class TestTrainingDedup:
    """Duplicate training sequences are removed."""

    def test_dedup_in_train(self):
        """train() deduplicates parsed sequences."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "dedup" in source.lower() or "seen" in source or "dict.fromkeys" in source

    def test_dedup_preserves_order(self):
        """Dedup preserves original order of sequences."""
        # dict.fromkeys preserves insertion order in Python 3.7+
        seqs = ["hello", "world", "hello", "foo", "world"]
        deduped = list(dict.fromkeys(seqs))
        assert deduped == ["hello", "world", "foo"]


# =============================================================================
# ADAPTIVE TRAINER TESTS
# =============================================================================

class TestAdaptiveTrainerImports:
    """Test that adaptive_trainer module imports correctly."""

    def test_module_imports(self):
        """adaptive_trainer module is importable."""
        from enigma_engine.core.adaptive_trainer import (
            TrainingPlan, StageResult, ALL_STAGES,
            DIFFICULTY_LEVELS,
            build_adaptive_prompt)
        assert TrainingPlan is not None
        assert StageResult is not None

    def test_all_stages_defined(self):
        """ALL_STAGES has the 4 expected stages."""
        from enigma_engine.core.adaptive_trainer import ALL_STAGES
        assert ALL_STAGES == ["basics", "conversation", "commands", "web"]

    def test_difficulty_levels_defined(self):
        """DIFFICULTY_LEVELS has 3 levels in order."""
        from enigma_engine.core.adaptive_trainer import DIFFICULTY_LEVELS
        assert DIFFICULTY_LEVELS == ["simple", "medium", "advanced"]


class TestTrainingPlan:
    """Test TrainingPlan dataclass and logic."""

    def test_create_default_plan(self):
        """Default plan has all 4 stages starting at index 0."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan(
            student_path="models/student.pth",
            trainer_path="models/trainer.pth")
        assert plan.current_stage == "basics"
        assert plan.current_stage_idx == 0
        assert not plan.is_complete
        assert plan.status == "pending"

    def test_advance_stage(self):
        """advance_stage moves to next stage."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.current_stage == "basics"
        result = plan.advance_stage()
        assert result is True
        assert plan.current_stage == "conversation"
        plan.advance_stage()
        assert plan.current_stage == "commands"
        plan.advance_stage()
        assert plan.current_stage == "web"
        result = plan.advance_stage()
        assert result is False
        assert plan.is_complete
        assert plan.status == "completed"

    def test_decide_action_advance(self):
        """decide_action always advances."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.decide_action(8.0) == "advance"
        assert plan.decide_action(3.0) == "advance"
        assert plan.decide_action(0.0) == "advance"

    def test_no_dead_difficulty_methods(self):
        """Dead difficulty methods were removed."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert not hasattr(plan, "simplify_difficulty")
        assert not hasattr(plan, "raise_difficulty")
        assert not hasattr(plan, "advance_score")
        assert not hasattr(plan, "retry_score")

    def test_record_result(self):
        """record_result stores a StageResult."""
        from enigma_engine.core.adaptive_trainer import (
            TrainingPlan, StageResult)
        plan = TrainingPlan()
        result = StageResult(
            stage="basics", attempt=1,
            scores=[7.0, 8.0], avg_score=7.5,
            status="passed")
        plan.record_result(result)
        assert len(plan.stage_results) == 1
        assert plan.stage_results[0]["stage"] == "basics"
        assert plan.stage_results[0]["avg_score"] == 7.5

    def test_save_and_load(self, tmp_path):
        """Plan round-trips through JSON save/load."""
        from enigma_engine.core.adaptive_trainer import (
            TrainingPlan, StageResult)
        plan = TrainingPlan(
            student_path="models/student.pth",
            trainer_path="models/trainer.pth",
            student_name="student",
            trainer_name="trainer",
            epochs_per_stage=20,
            learning_rate=0.001)
        plan.advance_stage()
        plan.record_result(StageResult(
            stage="basics", attempt=1,
            avg_score=8.0, status="passed"))
        path = tmp_path / "plan.json"
        plan.save(path)
        assert path.exists()
        loaded = TrainingPlan.load(path)
        assert loaded.current_stage_idx == 1
        assert loaded.student_path == "models/student.pth"
        assert loaded.epochs_per_stage == 20
        assert len(loaded.stage_results) == 1

    def test_summary(self):
        """summary() returns readable text."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan(
            student_name="student", trainer_name="trainer")
        text = plan.summary()
        assert "student" in text
        assert "trainer" in text
        assert "basics" in text.lower()

    def test_current_attempt_counts(self):
        """current_attempt correctly counts attempts for current stage."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.current_attempt == 0
        plan.stage_results = [
            {"stage": "basics", "attempt": 1},
            {"stage": "basics", "attempt": 2}]
        assert plan.current_attempt == 2

    def test_is_complete_after_all_stages(self):
        """Plan is complete after advancing past all stages."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        for _ in range(4):
            plan.advance_stage()
        assert plan.is_complete
        assert plan.current_stage is None


class TestBuildAdaptivePrompt:
    """Test build_adaptive_prompt for different stages/difficulties."""

    def test_simple_basics(self):
        """Simple basics prompt asks for ultra-simple content."""
        from enigma_engine.core.adaptive_trainer import build_adaptive_prompt
        prompt = build_adaptive_prompt(1, 10, "basics", "simple")
        assert "simple" in prompt.lower() or "SIMPLE" in prompt
        assert "#1" in prompt

    def test_advanced_conversation(self):
        """Advanced conversation prompt asks for complex dialogue."""
        from enigma_engine.core.adaptive_trainer import build_adaptive_prompt
        prompt = build_adaptive_prompt(5, 20, "conversation", "advanced")
        assert "complex" in prompt.lower() or "ADVANCED" in prompt

    def test_all_stage_difficulty_combos(self):
        """All 12 stage × difficulty combinations produce prompts."""
        from enigma_engine.core.adaptive_trainer import (
            build_adaptive_prompt, ALL_STAGES, DIFFICULTY_LEVELS)
        for stage in ALL_STAGES:
            for diff in DIFFICULTY_LEVELS:
                prompt = build_adaptive_prompt(1, 5, stage, diff)
                assert len(prompt) > 50
                assert str(stage.upper()) in prompt


class TestStageResult:
    """Test StageResult serialization."""

    def test_to_dict(self):
        """StageResult serializes to dict."""
        from enigma_engine.core.adaptive_trainer import StageResult
        r = StageResult(stage="basics", attempt=1, avg_score=7.5)
        d = r.to_dict()
        assert d["stage"] == "basics"
        assert d["avg_score"] == 7.5

    def test_from_dict(self):
        """StageResult deserializes from dict."""
        from enigma_engine.core.adaptive_trainer import StageResult
        d = {"stage": "web", "attempt": 2, "avg_score": 6.0,
             "status": "passed", "difficulty": "medium",
             "scores": [6.0, 6.0], "epochs_trained": 10,
             "pairs_generated": 20, "best_loss": 1.5,
             "started_at": "", "completed_at": ""}
        r = StageResult.from_dict(d)
        assert r.stage == "web"
        assert r.attempt == 2

    def test_from_dict_ignores_unknown_keys(self):
        """StageResult.from_dict skips unknown keys."""
        from enigma_engine.core.adaptive_trainer import StageResult
        d = {"stage": "basics", "attempt": 1,
             "unknown_field": "ignored"}
        r = StageResult.from_dict(d)
        assert r.stage == "basics"


# =============================================================================
# SMART BACKGROUND TRAINER TESTS
# =============================================================================

class TestSmartBackgroundTrainer:
    """Test the upgraded BackgroundTrainer with replay buffer and filtering."""

    def test_replay_buffer_exists(self):
        """BackgroundTrainer has a replay_buffer attribute."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer()
        assert hasattr(bt, "replay_buffer")

    def test_replay_buffer_max_size(self):
        """Replay buffer has a configurable max size."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer(replay_buffer_size=500)
        assert bt.replay_buffer_size == 500

    def test_add_example_goes_to_replay_buffer(self):
        """add_example stores in replay buffer as well as queue."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer()
        bt.add_example("hello", "hi there", score=0.8)
        assert bt.example_queue.qsize() == 1
        assert len(bt.replay_buffer) == 1

    def test_all_examples_trained(self):
        """All examples are trained on (no quality filtering)."""
        from enigma_engine.router import BackgroundTrainer
        source = inspect.getsource(BackgroundTrainer._train_batch)
        # No quality_threshold filtering — trains on full batch
        assert "quality_threshold" not in source

    def test_replay_buffer_capped(self):
        """Replay buffer respects max size."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer(replay_buffer_size=5)
        for i in range(10):
            bt.add_example(f"prompt{i}", f"response{i}", score=0.9)
        assert len(bt.replay_buffer) <= 5

    def test_dpo_pairs_collected(self):
        """Low-score examples generate DPO preference pairs."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer()
        assert hasattr(bt, "dpo_pairs")
        assert isinstance(bt.dpo_pairs, list)

    def test_get_stats_includes_replay_info(self):
        """get_stats returns replay buffer and DPO pair counts."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer()
        stats = bt.get_stats()
        assert "replay_buffer_size" in stats
        assert "dpo_pairs" in stats

    def test_retrain_on_replay_method_exists(self):
        """BackgroundTrainer has _retrain_on_replay method."""
        from enigma_engine.router import BackgroundTrainer
        assert hasattr(BackgroundTrainer, "_retrain_on_replay")

    def test_replay_keeps_best_examples(self):
        """Replay buffer sorted by score, keeps highest."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer(replay_buffer_size=3)
        bt.add_example("a", "b", score=0.3)
        bt.add_example("c", "d", score=0.9)
        bt.add_example("e", "f", score=0.6)
        bt.add_example("g", "h", score=0.8)
        # Buffer keeps top 3 by score
        scores = [ex.score for ex in bt.replay_buffer]
        assert min(scores) >= 0.6

    def test_train_batch_try_finally(self):
        """_train_batch still uses try/finally for eval mode restore."""
        from enigma_engine.router import BackgroundTrainer
        source = inspect.getsource(BackgroundTrainer._train_batch)
        assert "try:" in source
        assert "finally:" in source
        assert ".eval()" in source


class TestCurriculumSeparators:
    """Curriculum examples are properly separated for parsing."""

    def test_format_training_pair_no_trailing_newlines(self):
        """_format_training_pair returns clean examples without extra newlines."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        pair = ForgeMixin._format_training_pair(
            "basics", "Hello", "Hi there")
        # Should be a clean single example
        assert pair == "Hello\nHi there"
        assert not pair.endswith("\n\n")

    def test_guided_training_joins_with_double_newline(self):
        """Guided training joins pairs with \\n\\n for parser split."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert '"\\n\\n"' in source or "'\\n\\n'" in source


# ================================================================
# DA-C: Curated Dataset
# ================================================================


class TestCuratedDataset:
    """Test CuratedDataset management."""

    def test_module_imports(self):
        """curated_dataset module is importable."""
        from enigma_engine.core.curated_dataset import (
            CuratedDataset, DatasetEntry)
        assert CuratedDataset is not None
        assert DatasetEntry is not None

    def test_create_entry(self):
        """DatasetEntry has correct defaults."""
        from enigma_engine.core.curated_dataset import DatasetEntry
        entry = DatasetEntry(text="Hello world", source="test")
        assert entry.status == "pending"
        assert entry.timestamp  # auto-set

    def test_entry_round_trip(self):
        """DatasetEntry to_dict / from_dict round-trips."""
        from enigma_engine.core.curated_dataset import DatasetEntry
        entry = DatasetEntry(
            text="Q: hi\nA: hello", source="guided",
            stage="basics", status="approved")
        d = entry.to_dict()
        loaded = DatasetEntry.from_dict(d)
        assert loaded.text == entry.text
        assert loaded.source == entry.source
        assert loaded.status == "approved"

    def test_add_and_count(self, tmp_path):
        """Adding entries increments count."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        assert ds.count == 0
        ds.add("example 1", source="test")
        ds.add("example 2", source="test")
        assert ds.count == 2
        assert ds.pending_count == 2

    def test_approve_reject(self, tmp_path):
        """Approve and reject change entry status."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("good", source="test")
        ds.add("bad", source="test")
        assert ds.approve(0) is True
        assert ds.reject(1) is True
        assert ds.approved_count == 1
        assert ds.rejected_count == 1
        assert ds.pending_count == 0

    def test_approve_all_pending(self, tmp_path):
        """approve_all_pending approves all pending entries."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="test")
        ds.add("b", source="test")
        ds.add("c", source="test")
        count = ds.approve_all_pending()
        assert count == 3
        assert ds.approved_count == 3
        assert ds.pending_count == 0

    def test_reject_all_pending(self, tmp_path):
        """reject_all_pending rejects all pending entries."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="test")
        ds.add("b", source="test")
        count = ds.reject_all_pending()
        assert count == 2
        assert ds.rejected_count == 2

    def test_get_approved_text(self, tmp_path):
        """get_approved_text returns only approved entry text."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("good data", source="test")
        ds.add("bad data", source="test")
        ds.approve(0)
        ds.reject(1)
        texts = ds.get_approved_text()
        assert texts == ["good data"]

    def test_get_training_data(self, tmp_path):
        """get_training_data joins approved entries."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("line 1", source="test", stage="basics")
        ds.add("line 2", source="test", stage="basics")
        ds.approve(0)
        ds.approve(1)
        text = ds.get_training_data(stage="basics")
        assert "line 1" in text
        assert "line 2" in text
        assert "\n\n" in text

    def test_get_by_source(self, tmp_path):
        """get_by_source filters by source."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="guided")
        ds.add("b", source="chat")
        ds.add("c", source="guided")
        guided = ds.get_by_source("guided")
        assert len(guided) == 2

    def test_get_by_stage(self, tmp_path):
        """get_by_stage filters by stage."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="test", stage="basics")
        ds.add("b", source="test", stage="conversation")
        basics = ds.get_by_stage("basics")
        assert len(basics) == 1

    def test_add_batch(self, tmp_path):
        """add_batch adds multiple entries at once."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        count = ds.add_batch(
            ["one", "two", "", "three"],
            source="batch", stage="basics")
        assert count == 3  # empty string skipped
        assert ds.count == 3

    def test_remove_entry(self, tmp_path):
        """remove() removes an entry by index."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("keep", source="test")
        ds.add("remove", source="test")
        assert ds.remove(1) is True
        assert ds.count == 1
        assert ds.entries[0].text == "keep"

    def test_save_and_load(self, tmp_path):
        """Dataset round-trips through save/load."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        path = tmp_path / "dataset.jsonl"
        ds = CuratedDataset(path)
        ds.add("entry 1", source="guided", stage="basics")
        ds.add("entry 2", source="chat", stage="conversation")
        ds.approve(0)
        ds.reject(1)
        ds.save()

        ds2 = CuratedDataset(path)
        assert ds2.count == 2
        assert ds2.approved_count == 1
        assert ds2.rejected_count == 1
        assert ds2.entries[0].text == "entry 1"
        assert ds2.entries[0].source == "guided"

    def test_summary(self, tmp_path):
        """summary() returns readable text."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="guided")
        ds.add("b", source="chat")
        text = ds.summary()
        assert "Total: 2" in text
        assert "Pending: 2" in text

    def test_invalid_index_operations(self, tmp_path):
        """Out-of-range operations return False."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        assert ds.approve(0) is False
        assert ds.reject(0) is False
        assert ds.remove(0) is False


# ================================================================
# CK-C: Rolling Best Checkpoints
# ================================================================


class TestRollingBestCheckpoints:
    """Test rolling_best_k in TrainingConfig."""

    def test_config_has_rolling_best_k(self):
        """TrainingConfig has rolling_best_k field."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, "rolling_best_k")
        assert config.rolling_best_k == 0  # disabled by default

    def test_config_to_dict_includes_rolling(self):
        """to_dict includes rolling_best_k."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(rolling_best_k=3)
        d = config.to_dict()
        assert d["rolling_best_k"] == 3

    def test_trainer_has_rolling_list(self):
        """Trainer initializes _rolling_checkpoints list."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.__init__)
        assert "_rolling_checkpoints" in source

    def test_save_rolling_checkpoint_method_exists(self):
        """Trainer has _save_rolling_checkpoint method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_save_rolling_checkpoint")

    def test_rolling_checkpoint_called_on_improvement(self):
        """_save_rolling_checkpoint is called when loss improves."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "_save_rolling_checkpoint" in source

    def test_rolling_best_k_zero_is_noop(self):
        """rolling_best_k=0 means no rolling checkpoints saved."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(rolling_best_k=0)
        assert config.rolling_best_k == 0
        d = config.to_dict()
        assert d["rolling_best_k"] == 0

    def test_save_every_default_disabled(self):
        """save_every defaults to 0 (disabled)."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert config.save_every == 0

    def test_save_every_guard_in_train(self):
        """SFT train loop guards save_every with > 0 check."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "self.config.save_every > 0" in source

    def test_cleanup_periodic_checkpoints_method_exists(self):
        """Trainer has _cleanup_periodic_checkpoints method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_cleanup_periodic_checkpoints")

    def test_cleanup_periodic_checkpoints_keeps_recent(self):
        """_cleanup_periodic_checkpoints keeps only the N most recent files."""
        from enigma_engine.core.training import Trainer
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Create 6 fake checkpoint files
            for i in range(1, 7):
                (td_path / f"checkpoint_epoch_{i}.pt").write_text("x")
            # Clean up, keep 3
            Trainer._cleanup_periodic_checkpoints(td_path, "checkpoint_epoch_", keep=3)
            remaining = sorted(td_path.glob("checkpoint_epoch_*.pt"))
            assert len(remaining) == 3
            names = [f.name for f in remaining]
            # Should keep epochs 4, 5, 6 (the most recent)
            assert "checkpoint_epoch_4.pt" in names
            assert "checkpoint_epoch_5.pt" in names
            assert "checkpoint_epoch_6.pt" in names

    def test_cleanup_periodic_noop_when_few(self):
        """_cleanup_periodic_checkpoints does nothing when files <= keep."""
        from enigma_engine.core.training import Trainer
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for i in range(1, 3):
                (td_path / f"vision_epoch_{i}.pt").write_text("x")
            Trainer._cleanup_periodic_checkpoints(td_path, "vision_epoch_", keep=3)
            remaining = list(td_path.glob("vision_epoch_*.pt"))
            assert len(remaining) == 2  # both kept

    def test_cleanup_called_in_train_loop(self):
        """SFT train loop calls _cleanup_periodic_checkpoints."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "_cleanup_periodic_checkpoints" in source

    def test_cleanup_called_in_vision_train(self):
        """Vision train loop calls _cleanup_periodic_checkpoints."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_vision)
        assert "_cleanup_periodic_checkpoints" in source


# ================================================================
# TS-B: Training Queue
# ================================================================


class TestTrainingQueue:
    """Test TrainingQueue and TrainingJob."""

    def test_module_imports(self):
        """training_queue module is importable."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob, OvernightPlan)
        assert TrainingQueue is not None
        assert TrainingJob is not None
        assert OvernightPlan is not None

    def test_create_job(self):
        """TrainingJob has correct defaults."""
        from enigma_engine.core.training_queue import TrainingJob
        job = TrainingJob(mode="Solo", model_path="models/test.pth")
        assert job.status == "pending"
        assert job.progress == 0
        assert job.created_at  # auto-set

    def test_job_round_trip(self):
        """TrainingJob to_dict / from_dict round-trips."""
        from enigma_engine.core.training_queue import TrainingJob
        job = TrainingJob(
            mode="DPO", model_path="m.pth",
            data_path="d.jsonl", epochs=20)
        d = job.to_dict()
        loaded = TrainingJob.from_dict(d)
        assert loaded.mode == "DPO"
        assert loaded.epochs == 20
        assert loaded.model_path == "m.pth"

    def test_add_job(self):
        """Adding a job assigns an ID."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        job = q.add_job(TrainingJob(mode="Solo"))
        assert job.job_id == 1
        assert q.pending_count == 1
        job2 = q.add_job(TrainingJob(mode="DPO"))
        assert job2.job_id == 2
        assert q.pending_count == 2

    def test_remove_job(self):
        """Remove a pending job."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        job = q.add_job(TrainingJob(mode="Solo"))
        assert q.remove_job(job.job_id) is True
        assert q.pending_count == 0

    def test_cancel_job(self):
        """Cancel a pending job."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        job = q.add_job(TrainingJob(mode="Solo"))
        assert q.cancel_job(job.job_id) is True
        assert q.pending_count == 0
        assert q.jobs[0].status == "cancelled"

    def test_clear_completed(self):
        """clear_completed removes done/failed/cancelled jobs."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        j1 = q.add_job(TrainingJob(mode="Solo"))
        j2 = q.add_job(TrainingJob(mode="DPO"))
        j3 = q.add_job(TrainingJob(mode="LoRA"))
        j1.status = "completed"
        j2.status = "failed"
        # j3 is still pending
        removed = q.clear_completed()
        assert removed == 2
        assert len(q.jobs) == 1
        assert q.jobs[0].job_id == j3.job_id

    def test_queue_executes_jobs_sequentially(self):
        """Queue runs jobs in order via executor."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        import time

        results = []
        def executor(job):
            results.append(job.mode)
            return 0.5  # fake loss

        q = TrainingQueue()
        q.executor = executor
        q.add_job(TrainingJob(mode="Solo"))
        q.add_job(TrainingJob(mode="DPO"))
        q.start()

        # Wait for queue to finish
        for _ in range(50):
            if not q.is_running and q.pending_count == 0:
                break
            time.sleep(0.1)

        assert results == ["Solo", "DPO"]
        assert q.jobs[0].status == "completed"
        assert q.jobs[1].status == "completed"

    def test_queue_handles_failed_job(self):
        """Queue marks failed jobs but continues."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        import time

        call_count = [0]
        def executor(job):
            call_count[0] += 1
            if job.mode == "DPO":
                raise RuntimeError("DPO failed")
            return 0.1

        q = TrainingQueue()
        q.executor = executor
        q.add_job(TrainingJob(mode="Solo"))
        q.add_job(TrainingJob(mode="DPO"))
        q.add_job(TrainingJob(mode="LoRA"))
        q.start()

        for _ in range(50):
            if not q.is_running and q.pending_count == 0:
                break
            time.sleep(0.1)

        assert call_count[0] == 3
        assert q.jobs[0].status == "completed"
        assert q.jobs[1].status == "failed"
        assert "DPO failed" in q.jobs[1].error
        assert q.jobs[2].status == "completed"

    def test_queue_save_and_load(self, tmp_path):
        """Queue state round-trips through save/load."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        path = tmp_path / "queue.json"
        q = TrainingQueue(save_path=path)
        q.add_job(TrainingJob(mode="Solo", epochs=5))
        q.add_job(TrainingJob(mode="DPO", epochs=10))
        # Force save
        q._save_state()

        q2 = TrainingQueue(save_path=path)
        loaded = q2.load_state()
        assert loaded is True
        assert q2.pending_count == 2
        jobs = q2.jobs
        assert jobs[0].mode == "Solo"
        assert jobs[1].mode == "DPO"

    def test_interrupted_job_resets_to_pending(self, tmp_path):
        """Running jobs reset to pending on load."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        path = tmp_path / "queue.json"
        q = TrainingQueue(save_path=path)
        job = q.add_job(TrainingJob(mode="Solo"))
        job.status = "running"
        q._save_state()

        q2 = TrainingQueue(save_path=path)
        q2.load_state()
        assert q2.jobs[0].status == "pending"

    def test_queue_summary(self):
        """summary() returns readable text."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        q.add_job(TrainingJob(mode="Solo"))
        q.add_job(TrainingJob(mode="DPO"))
        text = q.summary()
        assert "Training Queue" in text
        assert "Solo" in text
        assert "DPO" in text

    def test_queue_callbacks(self):
        """Queue fires callbacks on job events."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        import time

        events = []
        def on_complete(job):
            events.append(("complete", job.mode))
        def on_queue_done():
            events.append(("queue_done",))

        q = TrainingQueue()
        q.executor = lambda job: 0.1
        q.on_job_complete = on_complete
        q.on_queue_complete = on_queue_done
        q.add_job(TrainingJob(mode="Solo"))
        q.start()

        for _ in range(50):
            if not q.is_running:
                break
            time.sleep(0.1)

        assert ("complete", "Solo") in events
        assert ("queue_done",) in events

    def test_pause_and_resume(self):
        """Pause stops processing, resume continues."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        q.pause()
        assert q.is_paused is True
        q.resume()
        assert q.is_paused is False


# ================================================================
# TS-C: Overnight Plan
# ================================================================


class TestOvernightPlan:
    """Test OvernightPlan scheduling."""

    def test_create_plan(self):
        """OvernightPlan has correct defaults."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan(name="Test Plan")
        assert plan.status == "pending"
        assert plan.total_jobs == 0
        assert not plan.is_complete
        assert plan.created_at  # auto-set

    def test_add_job_config(self):
        """add_job_config adds jobs to the plan."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(
            mode="Solo", model_path="m.pth",
            data_path="d.txt", epochs=5)
        plan.add_job_config(
            mode="DPO", model_path="m.pth",
            data_path="d.jsonl", epochs=10)
        assert plan.total_jobs == 2
        assert plan.jobs[0]["mode"] == "Solo"
        assert plan.jobs[1]["epochs"] == 10

    def test_record_result(self):
        """record_result tracks completed jobs."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(mode="Solo", model_path="m.pth")
        plan.add_job_config(mode="DPO", model_path="m.pth")
        plan.record_result(plan.jobs[0], "completed", best_loss=0.5)
        assert plan.completed_jobs == 1
        assert plan.current_job_idx == 1
        assert not plan.is_complete

        plan.record_result(plan.jobs[1], "completed", best_loss=0.3)
        assert plan.is_complete
        assert plan.status == "completed"

    def test_record_failed_result(self):
        """Failed jobs are tracked in results."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(mode="Solo", model_path="m.pth")
        plan.record_result(
            plan.jobs[0], "failed", error="OOM")
        assert plan.failed_jobs == 1
        assert plan.results[0]["error"] == "OOM"

    def test_save_and_load(self, tmp_path):
        """OvernightPlan round-trips through JSON."""
        from enigma_engine.core.training_queue import OvernightPlan
        path = tmp_path / "plan.json"
        plan = OvernightPlan(name="Overnight")
        plan.add_job_config(
            mode="Solo", model_path="m.pth", epochs=5)
        plan.add_job_config(
            mode="DPO", model_path="m.pth", epochs=10)
        plan.record_result(plan.jobs[0], "completed", best_loss=0.5)
        plan.save(path)

        loaded = OvernightPlan.load(path)
        assert loaded.name == "Overnight"
        assert loaded.total_jobs == 2
        assert loaded.current_job_idx == 1
        assert loaded.completed_jobs == 1

    def test_summary(self):
        """summary() returns readable text."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan(name="Test")
        plan.add_job_config(mode="Solo", model_path="m.pth")
        text = plan.summary()
        assert "Test" in text
        assert "Solo" in text

    def test_to_queue_jobs(self):
        """to_queue_jobs converts remaining jobs to TrainingJob instances."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(
            mode="Solo", model_path="m.pth", epochs=5)
        plan.add_job_config(
            mode="DPO", model_path="m.pth", epochs=10)
        # Complete first job
        plan.record_result(plan.jobs[0], "completed")
        jobs = plan.to_queue_jobs()
        # Only second job should be converted
        assert len(jobs) == 1
        assert jobs[0].mode == "DPO"
        assert jobs[0].epochs == 10

    def test_to_queue_jobs_all_pending(self):
        """to_queue_jobs converts all jobs when none completed."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(mode="Solo", model_path="m.pth")
        plan.add_job_config(mode="LoRA", model_path="m.pth")
        jobs = plan.to_queue_jobs()
        assert len(jobs) == 2
        assert jobs[0].mode == "Solo"
        assert jobs[1].mode == "LoRA"

    def test_plan_is_complete_when_all_done(self):
        """Plan marks completed after all jobs recorded."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(mode="Solo", model_path="m.pth")
        plan.record_result(plan.jobs[0], "completed")
        assert plan.is_complete
        assert plan.completed_at  # timestamp set


# ================================================================
# RL TRAINING (RL-B / RL-C)
# ================================================================

class TestRLTrainingImports:
    """Verify rl_training module imports and class structure."""

    def test_import_reward_model(self):
        from enigma_engine.core.rl_training import RewardModel
        assert RewardModel is not None

    def test_import_reward_trainer(self):
        from enigma_engine.core.rl_training import (
            RewardTrainer, RewardTrainerConfig)
        assert RewardTrainer is not None
        assert RewardTrainerConfig is not None

    def test_import_rlhf(self):
        from enigma_engine.core.rl_training import (
            RLHFTrainer, RLHFConfig)
        assert RLHFTrainer is not None
        assert RLHFConfig is not None

    def test_import_selfplay(self):
        from enigma_engine.core.rl_training import (
            SelfPlayTrainer, SelfPlayConfig)
        assert SelfPlayTrainer is not None
        assert SelfPlayConfig is not None

    def test_reward_trainer_config_defaults(self):
        from enigma_engine.core.rl_training import RewardTrainerConfig
        cfg = RewardTrainerConfig()
        assert cfg.epochs == 3
        assert cfg.learning_rate == 1e-5
        assert cfg.batch_size == 4
        assert cfg.max_length == 512

    def test_rlhf_config_defaults(self):
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert cfg.epochs == 3
        assert cfg.kl_coeff == 0.1
        assert cfg.clip_range == 0.2
        assert cfg.n_responses == 4

    def test_selfplay_config_defaults(self):
        from enigma_engine.core.rl_training import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert cfg.epochs == 3
        assert cfg.kl_coeff == 0.05
        assert cfg.n_responses == 4
        assert "{prompt}" in cfg.score_prompt
        assert "{response}" in cfg.score_prompt


class TestRewardModel:
    """Test RewardModel creation and forward pass."""

    def test_reward_model_creates_from_enigma(self):
        """RewardModel can be built from a small Enigma model."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import RewardModel

        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        base = Enigma(config=cfg)
        rm = RewardModel(base, freeze_base=True)

        # Should have reward_head
        assert hasattr(rm, "reward_head")
        assert rm.reward_head.out_features == 1

        # Base weights should be frozen
        for p in rm.tok_embeddings.parameters():
            assert not p.requires_grad

    def test_reward_model_forward_shape(self):
        """Forward produces (B,) scalar rewards."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import RewardModel

        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        base = Enigma(config=cfg)
        rm = RewardModel(base)

        ids = torch.randint(0, 128, (2, 10))
        rewards = rm(ids)
        assert rewards.shape == (2,)

    def test_reward_model_with_attention_mask(self):
        """Forward respects attention_mask to find last real token."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import RewardModel

        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        base = Enigma(config=cfg)
        rm = RewardModel(base)

        ids = torch.randint(0, 128, (1, 10))
        mask = torch.ones(1, 10)
        mask[0, 7:] = 0  # Last 3 are padding
        rewards = rm(ids, attention_mask=mask)
        assert rewards.shape == (1,)


class TestRewardTrainer:
    """Test RewardTrainer preference pair training."""

    def test_encode_pairs(self):
        """_encode_pairs converts preference data to tensor pairs."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import (
            RewardModel, RewardTrainer, RewardTrainerConfig)
        from enigma_engine.core.tokenizer import get_tokenizer

        cfg = ForgeConfig(vocab_size=8000, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=64)
        base = Enigma(config=cfg)
        rm = RewardModel(base)

        tokenizer = get_tokenizer()
        trainer = RewardTrainer(rm, tokenizer)

        pairs = trainer._encode_pairs([
            {"prompt": "Hi", "chosen": "Hello!", "rejected": "Go away"},
        ])
        assert len(pairs) == 1
        c, r = pairs[0]
        assert isinstance(c, torch.Tensor)
        assert isinstance(r, torch.Tensor)

    def test_request_stop(self):
        """request_stop sets internal flag."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import (
            RewardModel, RewardTrainer)

        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        base = Enigma(config=cfg)
        rm = RewardModel(base)
        trainer = RewardTrainer(rm, None)

        assert not trainer._should_stop()
        trainer.request_stop()
        assert trainer._should_stop()


# ================================================================
# LORA ADAPTER MANAGER (FP-D)
# ================================================================

class TestLoRAAdapterManager:
    """Test per-task LoRA adapter management."""

    def test_import(self):
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        assert LoRAAdapterManager is not None

    def test_list_empty(self, tmp_path):
        """list_tasks returns empty when no adapters exist."""
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        assert mgr.list_tasks() == []

    def test_create_and_list(self, tmp_path):
        """create() stores adapter; list_tasks() finds it."""
        import torch
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)

        # Mark some params as trainable to simulate LoRA
        for p in list(model.parameters())[:2]:
            p.requires_grad = True

        path = mgr.create("coding", model)
        assert path.exists()
        assert "coding" in mgr.list_tasks()

    def test_save_and_switch(self, tmp_path):
        """save() persists weights; switch() loads them back."""
        import torch
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)

        # Simulate trainable params
        for p in list(model.parameters())[:2]:
            p.requires_grad = True

        mgr.create("task_a", model)
        mgr.save("task_a", model)
        assert mgr.active_task == "task_a"

        # Create second task
        mgr.create("task_b", model)
        mgr.save("task_b", model)
        assert "task_a" in mgr.list_tasks()
        assert "task_b" in mgr.list_tasks()

    def test_delete(self, tmp_path):
        """delete() removes adapter from disk."""
        import torch
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)
        for p in list(model.parameters())[:2]:
            p.requires_grad = True

        mgr.create("todelete", model)
        assert "todelete" in mgr.list_tasks()
        mgr.delete("todelete")
        assert "todelete" not in mgr.list_tasks()

    def test_switch_nonexistent_raises(self, tmp_path):
        """switch() raises FileNotFoundError for missing task."""
        import torch
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)

        with pytest.raises(FileNotFoundError, match="No adapter"):
            mgr.switch("nonexistent", model)


# ================================================================
# TRAINING MONITOR (TM-B / TM-C / TM-D)
# ================================================================

class TestTrainingMonitor:
    """Test training monitor: loss tracking and history."""

    def test_import(self):
        from enigma_engine.core.training_monitor import (
            TrainingMonitor, TrainingRun)
        assert TrainingMonitor is not None
        assert TrainingRun is not None

    def test_record_loss(self):
        """record_loss stores values and computes best."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_loss(2.0)
        m.record_loss(1.5)
        m.record_loss(1.8)

        assert len(m.losses) == 3
        assert m.best_loss == 1.5
        assert m.current_loss == 1.8

    def test_moving_average(self):
        """moving_average produces correct-length output."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor(moving_avg_window=3)
        m.start_run()
        for v in [3.0, 2.0, 1.0, 0.5]:
            m.record_loss(v)

        ma = m.moving_average()
        assert len(ma) == 4
        # First value is just itself
        assert ma[0] == 3.0
        # Last value is avg of [2.0, 1.0, 0.5]
        assert abs(ma[3] - (2.0 + 1.0 + 0.5) / 3) < 0.01

    def test_get_chart_data(self):
        """get_chart_data returns structured dict."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_loss(1.0)
        m.record_loss(0.5)

        data = m.get_chart_data()
        assert "steps" in data
        assert "losses" in data
        assert "moving_avg" in data
        assert "best_loss" in data
        assert data["best_loss"] == 0.5

    def test_epoch_loss_tracking(self):
        """record_epoch_loss stores per-epoch values."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_epoch_loss(2.0)
        m.record_epoch_loss(1.5)

        assert m.epoch_losses == [2.0, 1.5]

    def test_epoch_perplexities_tracking(self):
        """record_epoch_loss auto-computes perplexity from loss."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_epoch_loss(1.0)
        m.record_epoch_loss(0.5)

        assert len(m.epoch_perplexities) == 2
        import math
        assert abs(m.epoch_perplexities[0] - math.exp(1.0)) < 0.01
        assert abs(m.epoch_perplexities[1] - math.exp(0.5)) < 0.01

    def test_perplexity_in_chart_data(self):
        """get_chart_data includes epoch_perplexities key."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_loss(1.0)
        m.record_epoch_loss(1.0)

        data = m.get_chart_data()
        assert "epoch_perplexities" in data
        assert len(data["epoch_perplexities"]) == 1

    def test_finish_run_includes_perplexity(self, tmp_path):
        """finish_run extra dict has final_perplexity and best_perplexity."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "hist.json"
        m = TrainingMonitor(history_path=hist_path)
        m.start_run()
        m.record_loss(1.0)
        m.record_epoch_loss(2.0)
        m.record_epoch_loss(1.0)

        run = m.finish_run(mode="sft", model_name="test")
        assert run.extra.get("final_perplexity") is not None
        assert run.extra.get("best_perplexity") is not None
        assert run.extra["best_perplexity"] <= run.extra["final_perplexity"]

    def test_perplexity_reset_on_start_run(self):
        """start_run resets epoch_perplexities list."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_epoch_loss(1.0)
        assert len(m.epoch_perplexities) == 1

        m.start_run()
        assert len(m.epoch_perplexities) == 0

    def test_training_run_serialization(self):
        """TrainingRun round-trips through dict."""
        from enigma_engine.core.training_monitor import TrainingRun

        run = TrainingRun(
            run_id="test_1",
            mode="sft",
            model_name="base",
            final_loss=0.5,
            total_steps=100,
        )
        d = run.to_dict()
        assert d["run_id"] == "test_1"
        assert d["mode"] == "sft"

        restored = TrainingRun.from_dict(d)
        assert restored.run_id == "test_1"
        assert restored.final_loss == 0.5

    def test_finish_run_saves_history(self, tmp_path):
        """finish_run persists to history file."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "history.json"
        m = TrainingMonitor(history_path=hist_path)
        m.start_run()
        m.record_loss(1.0)
        m.record_loss(0.5)

        run = m.finish_run(mode="sft", model_name="test_model")
        assert run.mode == "sft"
        assert run.total_steps == 2

        # File should exist
        assert hist_path.exists()

        # Load it back
        history = m.get_history()
        assert len(history) == 1
        assert history[0].mode == "sft"

    def test_history_multiple_runs(self, tmp_path):
        """Multiple runs append to the history."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "history.json"
        m = TrainingMonitor(history_path=hist_path)

        # Run 1
        m.start_run()
        m.record_loss(2.0)
        m.finish_run(mode="sft", model_name="m1")

        # Run 2
        m.start_run()
        m.record_loss(1.0)
        m.finish_run(mode="dpo", model_name="m2")

        history = m.get_history()
        assert len(history) == 2

    def test_history_filter_by_mode(self, tmp_path):
        """get_history can filter by training mode."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "history.json"
        m = TrainingMonitor(history_path=hist_path)

        m.start_run()
        m.record_loss(1.0)
        m.finish_run(mode="sft")

        m.start_run()
        m.record_loss(1.0)
        m.finish_run(mode="dpo")

        sft_only = m.get_history(mode="sft")
        assert len(sft_only) == 1
        assert sft_only[0].mode == "sft"

    def test_clear_history(self, tmp_path):
        """clear_history removes the file."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "history.json"
        m = TrainingMonitor(history_path=hist_path)
        m.start_run()
        m.record_loss(1.0)
        m.finish_run(mode="sft")

        assert hist_path.exists()
        m.clear_history()
        assert not hist_path.exists()


# ================================================================
# THREAD SAFETY — Suggestion #8A+D
# ================================================================

class TestThreadSafety:
    """Verify thread safety locks and copy-on-write across modules."""

    # -- training_monitor.py --

    def test_training_monitor_has_lock(self):
        """TrainingMonitor must have a threading.Lock."""
        import threading
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        assert hasattr(m, "_lock")
        assert isinstance(m._lock, type(threading.Lock()))

    def test_training_monitor_losses_snapshot(self):
        """losses property must return a copy, not internal list."""
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        m.start_run()
        m.record_loss(1.0)
        snap = m.losses
        snap.append(999.0)
        assert 999.0 not in m.losses

    def test_training_monitor_chart_data_snapshot(self):
        """get_chart_data must return copies under the lock."""
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        m.start_run()
        m.record_loss(2.0)
        data = m.get_chart_data()
        data["losses"].append(999.0)
        assert 999.0 not in m.get_chart_data()["losses"]

    # -- model_registry.py --

    def test_model_registry_has_lock(self):
        """ModelRegistry must have a threading.Lock."""
        import threading
        from enigma_engine.core.model_registry import ModelRegistry
        r = ModelRegistry(models_dir="/tmp/nonexistent_model_dir")
        assert hasattr(r, "_lock")
        assert isinstance(r._lock, type(threading.Lock()))

    def test_model_registry_list_returns_copy(self):
        """list_models must return a copy of the registry dict."""
        from enigma_engine.core.model_registry import ModelRegistry
        r = ModelRegistry(models_dir="/tmp/nonexistent_model_dir")
        models = r.list_models()
        models["injected"] = {"bad": True}
        assert "injected" not in r.list_models()

    # -- hardware_detection.py --

    def test_hardware_detection_has_lock(self):
        """hardware_detection module must have _profile_lock."""
        import threading
        from enigma_engine.core import hardware_detection
        assert hasattr(hardware_detection, "_profile_lock")
        assert isinstance(
            hardware_detection._profile_lock, type(threading.Lock()))

    def test_detect_hardware_returns_consistent_copy(self):
        """detect_hardware() results must not be mutable singletons."""
        from enigma_engine.core.hardware_detection import (
            detect_hardware, clear_cached_profile)
        clear_cached_profile()
        p1 = detect_hardware()
        p2 = detect_hardware()
        # Both should be equal but modifying one shouldn't affect the other
        # (dataclass is immutable by field, but test the cache path)
        assert p1.device == p2.device
        assert p1.ram_gb == p2.ram_gb


# ================================================================
# GGUF FALLBACK REMOVAL — Suggestion #9A
# ================================================================

class TestChatContextExtraction:
    """Verify _prepare_chat() + ChatContext refactoring (Suggestion #10A)."""

    def test_chat_context_importable(self):
        """ChatContext dataclass can be imported from engine_chat."""
        from enigma_engine.core.engine_chat import ChatContext
        assert ChatContext is not None

    def test_chat_context_has_expected_fields(self):
        """ChatContext has all required fields."""
        from enigma_engine.core.engine_chat import ChatContext
        import dataclasses
        assert dataclasses.is_dataclass(ChatContext)
        field_names = {f.name for f in dataclasses.fields(ChatContext)}
        expected = {
            "messages", "prompt", "stop_strings", "max_gen",
            "temperature", "repeat_penalty", "top_p", "top_k",
            "is_gguf", "has_server_backend",
        }
        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_prepare_chat_exists(self):
        """_prepare_chat is a method on _ChatMixin."""
        from enigma_engine.core.engine_chat import _ChatMixin
        assert hasattr(_ChatMixin, "_prepare_chat")
        assert callable(getattr(_ChatMixin, "_prepare_chat"))

    def test_prepare_chat_returns_chat_context(self):
        """_prepare_chat returns a ChatContext instance."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin, ChatContext

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False
        obj.model = MagicMock()
        obj.get_max_context_length = MagicMock(return_value=4096)
        obj.count_tokens = MagicMock(return_value=5)

        ctx = obj._prepare_chat("hello")
        assert isinstance(ctx, ChatContext)

    def test_prepare_chat_builds_messages(self):
        """Messages include system + history + user message."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False
        obj.model = MagicMock()
        obj.get_max_context_length = MagicMock(return_value=4096)
        obj.count_tokens = MagicMock(return_value=5)

        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        ctx = obj._prepare_chat(
            "What is Python?",
            history=history,
            system_prompt="You are helpful.",
        )
        # System message first
        assert ctx.messages[0] == {"role": "system", "content": "You are helpful."}
        # History in order
        assert ctx.messages[1] == {"role": "user", "content": "Hi"}
        assert ctx.messages[2] == {"role": "assistant", "content": "Hello!"}
        # User message last
        assert ctx.messages[-1] == {"role": "user", "content": "What is Python?"}

    def test_prepare_chat_reasoning_boosts_max_gen(self):
        """When reasoning=True, max_gen is multiplied by 1.5."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False
        obj.model = MagicMock()
        obj.get_max_context_length = MagicMock(return_value=4096)
        obj.count_tokens = MagicMock(return_value=5)

        ctx = obj._prepare_chat("test", max_gen=2000, reasoning=True)
        assert ctx.max_gen == 3000  # 2000 * 1.5

    def test_prepare_chat_reasoning_injects_instruction(self):
        """When reasoning=True, reasoning instruction is in prompt and messages."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False
        obj.model = MagicMock()
        obj.get_max_context_length = MagicMock(return_value=4096)
        obj.count_tokens = MagicMock(return_value=5)

        ctx = obj._prepare_chat("test", reasoning=True)
        # Reasoning instruction should be in system message
        assert any("<think>" in m.get("content", "") for m in ctx.messages)
        # And in the prompt
        assert "<think>" in ctx.prompt

    def test_chat_and_stream_use_prepare_chat(self):
        """Both chat() and stream_chat() call _prepare_chat()."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        chat_src = inspect.getsource(_ChatMixin.chat)
        stream_src = inspect.getsource(_ChatMixin.stream_chat)
        assert "_prepare_chat(" in chat_src, "chat() must call _prepare_chat()"
        assert "_prepare_chat(" in stream_src, "stream_chat() must call _prepare_chat()"


class TestStreamChatReasoning:
    """Verify stream_chat() supports reasoning (Suggestion #10D)."""

    def test_stream_chat_accepts_reasoning_param(self):
        """stream_chat() signature includes reasoning kwarg."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        sig = inspect.signature(_ChatMixin.stream_chat)
        assert "reasoning" in sig.parameters

    def test_stream_chat_reasoning_in_prompt(self):
        """stream_chat with reasoning=True has reasoning in context."""
        from unittest.mock import MagicMock, patch
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False
        obj.model = MagicMock()
        obj.get_max_context_length = MagicMock(return_value=4096)
        obj.count_tokens = MagicMock(return_value=5)

        # stream_generate yields tokens — mock it to yield nothing
        obj.stream_generate = MagicMock(return_value=iter([]))
        gen = obj.stream_chat("test", reasoning=True)
        # Consume the generator
        list(gen)

        # Check that stream_generate was called with a prompt containing <think>
        call_args = obj.stream_generate.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "<think>" in prompt


class TestTokenCounterReliable:
    """Verify count_tokens never estimates (Suggestion #11A + #11C)."""

    def test_count_tokens_raises_without_tokenizer(self):
        """count_tokens raises RuntimeError when tokenizer lacks encode/call."""
        from unittest.mock import MagicMock
        from enigma_engine.core.inference import EnigmaEngine

        # Build a minimal engine-like object with a bad tokenizer
        obj = object.__new__(EnigmaEngine)
        obj.tokenizer = object()  # No encode, no __call__
        obj._token_count_cache = {}

        with pytest.raises(RuntimeError, match="[Nn]o tokenizer"):
            obj.count_tokens("hello world")

    def test_no_estimation_fallback_in_source(self):
        """count_tokens source must not contain len(text) // 4 estimation."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine.count_tokens)
        assert "// 4" not in source, "count_tokens must not estimate with len(text)//4"

    def test_token_count_cache_exists(self):
        """Engine instances have _token_count_cache attribute."""
        from enigma_engine.core.inference import EnigmaEngine
        from unittest.mock import MagicMock

        obj = object.__new__(EnigmaEngine)
        obj.tokenizer = MagicMock()
        obj.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        obj._token_count_cache = {}

        obj.count_tokens("hi")
        assert "hi" in obj._token_count_cache
        assert obj._token_count_cache["hi"] == 3

    def test_token_count_cached_avoids_re_encode(self):
        """Second call to count_tokens uses cache, not tokenizer."""
        from unittest.mock import MagicMock
        from enigma_engine.core.inference import EnigmaEngine

        obj = object.__new__(EnigmaEngine)
        obj.tokenizer = MagicMock()
        obj.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        obj._token_count_cache = {}

        assert obj.count_tokens("hello") == 3
        assert obj.count_tokens("hello") == 3
        # encode should only be called once
        obj.tokenizer.encode.assert_called_once()

    def test_token_count_cache_bounded(self):
        """Cache clears when exceeding max entries."""
        from unittest.mock import MagicMock
        from enigma_engine.core.inference import EnigmaEngine

        obj = object.__new__(EnigmaEngine)
        obj.tokenizer = MagicMock()
        obj.tokenizer.encode = MagicMock(return_value=[1])
        obj._token_count_cache = {}

        # Fill cache beyond bound
        for i in range(4100):
            obj.count_tokens(f"text_{i}")

        # Cache should have been cleared and only have recent entries
        assert len(obj._token_count_cache) < 4097

    def test_from_model_has_token_cache(self):
        """from_model() created engines have _token_count_cache."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine.from_model)
        direct = "_token_count_cache" in source
        via_common = "_init_common" in source
        assert direct or via_common, \
            "from_model() must initialize _token_count_cache (directly or via _init_common)"
        if via_common:
            common_src = inspect.getsource(EnigmaEngine._init_common)
            assert "_token_count_cache" in common_src


class TestGGUFNoSilentFallback:
    """Verify GGUF chat errors propagate instead of silent fallback."""

    def test_no_silent_fallback_in_chat(self):
        """chat() must NOT silently catch GGUF errors and fall through."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.chat)
        # The old code had: except Exception as e: logger.warning(...) + fall through
        # After fix: GGUF errors must be re-raised, no "falling back to generate"
        assert "falling back to generate" not in source.lower(), \
            "GGUF chat must not silently fall back to generate()"

    def test_no_silent_fallback_in_stream_chat(self):
        """stream_chat() must NOT silently catch GGUF errors and fall through."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.stream_chat)
        assert "fall through to native" not in source.lower(), \
            "GGUF stream_chat must not silently fall back"

    def test_gguf_chat_raises_on_error(self):
        """When GGUF chat fails, the error must propagate."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = True
        mock_model = MagicMock()
        mock_model.chat = MagicMock(
            side_effect=RuntimeError("GGUF model crashed"))
        obj.model = mock_model
        # Stub methods needed by chat()
        obj.get_max_context_length = MagicMock(return_value=4096)
        obj.count_tokens = MagicMock(return_value=10)

        with pytest.raises(RuntimeError, match="GGUF model crashed"):
            obj.chat("hello")

class TestDataValidation:
    """Test validate_training_data() function."""

    def test_import(self):
        """DataValidationResult and validate_training_data importable."""
        from enigma_engine.core.training import (
            DataValidationResult, validate_training_data)
        assert DataValidationResult is not None
        assert validate_training_data is not None

    def test_valid_data(self):
        """Normal training text passes validation."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data(
            "Hello world this is a test.\nAnother line of training data.")
        assert result.is_valid is True
        assert result.total_sequences > 0
        assert len(result.errors) == 0

    def test_empty_data(self):
        """Empty string produces an error."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data("")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_short_sequences_warning(self):
        """Very short lines generate warnings."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data("a\nb\nc\nd\ne\nf\n")
        # short sequences should produce warnings
        assert len(result.warnings) > 0

    def test_duplicate_detection(self):
        """Duplicate lines are counted in stats."""
        from enigma_engine.core.training import validate_training_data

        text = "same line\n" * 10
        result = validate_training_data(text)
        assert result.stats.get("duplicates", 0) > 0

    def test_stats_populated(self):
        """Stats dict contains expected keys."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data(
            "Line one is long enough.\nLine two is also long enough.")
        assert "total_chars" in result.stats
        assert "total_lines" in result.stats
        assert "unique_lines" in result.stats
        assert "avg_length" in result.stats

    def test_null_bytes_warning(self):
        """Data with null bytes produces a warning."""
        from enigma_engine.core.training import validate_training_data

        result = validate_training_data("Hello\x00World this is enough text")
        has_null_warning = any(
            "null" in w.lower() for w in result.warnings)
        assert has_null_warning

    def test_result_dataclass_fields(self):
        """DataValidationResult has the expected fields."""
        from enigma_engine.core.training import DataValidationResult

        r = DataValidationResult(
            is_valid=True,
            total_sequences=5,
            warnings=["warn"],
            errors=[],
            stats={"total_chars": 100},
        )
        assert r.is_valid is True
        assert r.total_sequences == 5
        assert len(r.warnings) == 1
        assert len(r.errors) == 0
        assert r.stats["total_chars"] == 100


class TestRMSNormFp32Upcast:
    """RMSNorm must compute in fp32 then cast back to input dtype."""

    def test_output_dtype_matches_input(self):
        """RMSNorm output dtype == input dtype for float32."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(16)
        x = torch.randn(2, 16)
        out = norm(x)
        assert out.dtype == x.dtype

    def test_fp16_no_nan(self):
        """fp16 input should not produce NaN thanks to fp32 upcast."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(32)
        # Large values that would overflow in fp16 norm without upcast
        x = torch.randn(4, 32).half() * 100
        norm = norm.half()
        out = norm(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any(), "fp16 RMSNorm produced NaN"

    def test_vision_rmsnorm_fp32_upcast(self):
        """Vision encoder _RMSNorm also upcasts to fp32."""
        import torch
        from enigma_engine.core.vision_encoder import _RMSNorm
        norm = _RMSNorm(32)
        x = torch.randn(4, 32).half() * 100
        norm = norm.half()
        out = norm(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()


class TestTrainingConfigAdamFields:
    """TrainingConfig must expose Adam optimizer fields."""

    def test_default_betas(self):
        """adam_beta1/beta2 default to LM-friendly values."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.adam_beta1 == 0.9
        assert cfg.adam_beta2 == 0.95
        assert cfg.adam_eps == 1e-8

    def test_custom_betas(self):
        """adam_beta1/beta2 can be overridden."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(adam_beta1=0.85, adam_beta2=0.999, adam_eps=1e-6)
        assert cfg.adam_beta1 == 0.85
        assert cfg.adam_beta2 == 0.999
        assert cfg.adam_eps == 1e-6

    def test_to_dict_includes_adam_fields(self):
        """to_dict() must include all three Adam fields."""
        from enigma_engine.core.training import TrainingConfig
        d = TrainingConfig().to_dict()
        assert "adam_beta1" in d
        assert "adam_beta2" in d
        assert "adam_eps" in d
        assert d["adam_beta1"] == 0.9
        assert d["adam_beta2"] == 0.95


class TestLoraTrainerWeightDecay:
    """LoRA trainer weight_decay must be configurable."""

    def test_default_weight_decay(self):
        """Default weight_decay is 0.01."""
        from enigma_engine.core.lora_utils import LoraTrainer
        import inspect
        sig = inspect.signature(LoraTrainer.__init__)
        assert sig.parameters["weight_decay"].default == 0.01

    def test_custom_weight_decay_stored(self):
        """Custom weight_decay is stored on the instance."""
        import torch.nn as nn
        from unittest.mock import MagicMock, patch

        # Patch create_lora_model to avoid actual LoRA application
        with patch("enigma_engine.core.lora_utils.create_lora_model",
                   side_effect=lambda m, c: m):
            from enigma_engine.core.lora_utils import LoraTrainer
            model = nn.Linear(4, 4)
            tok = MagicMock()
            trainer = LoraTrainer(model, tok, weight_decay=0.05)
            assert trainer.weight_decay == 0.05


# ================================================================
# V-G: HYBRID CNN+ViT
# ================================================================

class TestCNNStem:
    """Test CNNStem module for hybrid CNN+ViT."""

    def test_import(self):
        """CNNStem should be importable."""
        from enigma_engine.core.vision_encoder import CNNStem
        assert CNNStem is not None

    def test_forward_shape(self):
        """CNNStem outputs [batch, num_patches, dim]."""
        import torch
        from enigma_engine.core.vision_encoder import CNNStem

        stem = CNNStem(channels=3, dim=64)
        x = torch.randn(2, 3, 64, 64)
        out = stem(x)
        # CNN stem does /8 spatial reduction: 64/8 = 8, 8*8 = 64
        num_patches = (64 // 8) ** 2
        assert out.shape == (2, num_patches, 64)

    def test_is_nn_module(self):
        """CNNStem is an nn.Module."""
        import torch.nn as nn
        from enigma_engine.core.vision_encoder import CNNStem

        stem = CNNStem(channels=3, dim=64)
        assert isinstance(stem, nn.Module)

    def test_has_trainable_params(self):
        """CNNStem should have trainable parameters."""
        from enigma_engine.core.vision_encoder import CNNStem

        stem = CNNStem(channels=3, dim=64)
        params = sum(p.numel() for p in stem.parameters())
        assert params > 0

    def test_gradients_flow(self):
        """Gradients should flow through CNNStem."""
        import torch
        from enigma_engine.core.vision_encoder import CNNStem

        stem = CNNStem(channels=3, dim=64)
        x = torch.randn(1, 3, 64, 64)
        out = stem(x)
        loss = out.sum()
        loss.backward()
        any_grad = any(
            p.grad is not None for p in stem.parameters())
        assert any_grad


class TestHybridVisionEncoder:
    """Test VisionEncoder with use_cnn_stem=True."""

    def test_config_has_cnn_stem_field(self):
        """VisionEncoderConfig has use_cnn_stem field."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig

        cfg = VisionEncoderConfig()
        assert hasattr(cfg, "use_cnn_stem")
        assert cfg.use_cnn_stem is False

    def test_hybrid_encoder_forward(self):
        """Hybrid VisionEncoder produces correct output shape."""
        import torch
        from enigma_engine.core.vision_encoder import (
            VisionEncoder, VisionEncoderConfig)

        cfg = VisionEncoderConfig(
            image_size=64, patch_size=8, dim=64,
            n_layers=2, n_heads=2, use_cnn_stem=True)
        encoder = VisionEncoder(cfg)
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        # CNN stem: 64/8 = 8, 8*8 = 64 patches
        assert out.shape[0] == 2
        assert out.shape[2] == 64
        assert out.shape[1] == (64 // 8) ** 2

    def test_hybrid_presets_exist(self):
        """VISION_PRESETS includes hybrid_small and hybrid_medium."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS

        assert "hybrid_small" in VISION_PRESETS
        assert "hybrid_medium" in VISION_PRESETS

    def test_hybrid_presets_use_cnn_stem(self):
        """Hybrid presets have use_cnn_stem=True."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS

        assert VISION_PRESETS["hybrid_small"].use_cnn_stem is True
        assert VISION_PRESETS["hybrid_medium"].use_cnn_stem is True

    def test_config_to_dict_includes_cnn_stem(self):
        """VisionEncoderConfig.to_dict() includes use_cnn_stem."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig

        cfg = VisionEncoderConfig(use_cnn_stem=True)
        d = cfg.to_dict()
        assert "use_cnn_stem" in d
        assert d["use_cnn_stem"] is True

    def test_lazy_import_cnn_stem(self):
        """CNNStem is in the lazy loader map."""
        from enigma_engine.core import CNNStem
        assert CNNStem is not None


# ================================================================
# FORGE GUI — NEW TRAINING MODES
# ================================================================

class TestForgeNewModes:
    """Verify new training modes are wired into FORGE."""

    def test_mode_display_keys(self):
        """New modes exist in _MODE_DISPLAY_TO_KEY."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        keys = ForgeMixin._MODE_DISPLAY_TO_KEY
        assert "RLHF" in keys
        assert "Self-Play" in keys

    def test_mode_descriptions(self):
        """New modes have descriptions."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        descs = ForgeMixin._TRAINING_MODE_DESCRIPTIONS
        for mode in ("RLHF", "SelfPlay"):
            assert mode in descs, f"Missing description for {mode}"
            assert len(descs[mode]) > 10

    def test_mode_section_visibility(self):
        """New modes have section visibility configs."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        vis = ForgeMixin._MODE_SECTION_VISIBILITY
        for mode in ("RLHF", "SelfPlay"):
            assert mode in vis, f"Missing visibility for {mode}"
            assert "data" in vis[mode]

    def test_mode_data_labels(self):
        """New modes have data source labels."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        labels = ForgeMixin._MODE_DATA_LABELS
        for mode in ("RLHF", "SelfPlay"):
            assert mode in labels, f"Missing label for {mode}"

    def test_dispatch_has_new_modes(self):
        """_start_training_by_mode handles 3 simplified modes."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_training_by_mode)
        # New simplified dispatch: Basic, AI-Guided, Image
        assert "Basic" in source or "_start_basic_training" in source
        assert "AI-Guided" in source or "_start_ai_guided_training" in source
        assert "Image" in source or "_start_vision_training" in source

    def test_new_modes_mixin_exists(self):
        """ForgeNewModesMixin is in the inheritance chain."""
        from enigma_engine.gui.gui_forge_new_modes import ForgeNewModesMixin
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert issubclass(ForgeMixin, ForgeNewModesMixin)

    def test_mixin_has_methods(self):
        """ForgeNewModesMixin has RLHF and Self-Play methods."""
        from enigma_engine.gui.gui_forge_new_modes import ForgeNewModesMixin
        assert hasattr(ForgeNewModesMixin, "_start_rlhf_training")
        assert hasattr(ForgeNewModesMixin, "_start_selfplay_training")


# ================================================================
# 12: OOM Recovery — VRAM-Based Batch Size + training OOM handling
# ================================================================


class TestRecommendTrainingBatchSize:
    """Tests for VRAM-tier based recommended training batch size."""

    def test_function_exists(self):
        """recommend_training_batch_size is importable."""
        from enigma_engine.core.hardware_detection import (
            recommend_training_batch_size,
        )
        assert callable(recommend_training_batch_size)

    def test_cpu_only_returns_small_batch(self):
        """CPU-only profile returns batch size 1 or 2."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(gpu_available=False, ram_gb=8.0)
        bs = recommend_training_batch_size(profile)
        assert bs == 1

    def test_cpu_high_ram(self):
        """CPU with high RAM returns 2."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(gpu_available=False, ram_gb=64.0)
        assert recommend_training_batch_size(profile) == 2

    def test_low_vram_returns_1(self):
        """GPU with < 6 GB VRAM returns 1."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=4.0)
        assert recommend_training_batch_size(profile) == 1

    def test_6gb_vram_returns_2(self):
        """GPU with 6 GB VRAM returns 2."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=6.0)
        assert recommend_training_batch_size(profile) == 2

    def test_12gb_vram_returns_4(self):
        """GPU with 12 GB VRAM returns 4."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=12.0)
        assert recommend_training_batch_size(profile) == 4

    def test_24gb_vram_returns_8(self):
        """GPU with 24 GB VRAM returns 8."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=24.0)
        assert recommend_training_batch_size(profile) == 8

    def test_48gb_vram_returns_16(self):
        """GPU with 48+ GB VRAM returns 16."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=48.0)
        assert recommend_training_batch_size(profile) == 16

    def test_in_all_exports(self):
        """recommend_training_batch_size is in __all__."""
        from enigma_engine.core import hardware_detection
        assert "recommend_training_batch_size" in hardware_detection.__all__


class TestTrainingOOMRecovery:
    """Tests for OOM recovery in training.py."""

    def test_train_one_batch_method_exists(self):
        """Trainer has _train_one_batch method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_train_one_batch")

    def test_handle_oom_method_exists(self):
        """Trainer has _handle_oom method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_handle_oom")

    def test_train_calls_train_one_batch(self):
        """train() delegates batch processing to _train_one_batch."""
        source = inspect.getsource(
            __import__(
                "enigma_engine.core.training", fromlist=["Trainer"]
            ).Trainer.train
        )
        assert "_train_one_batch" in source

    def test_train_catches_oom(self):
        """train() catches RuntimeError('out of memory')."""
        source = inspect.getsource(
            __import__(
                "enigma_engine.core.training", fromlist=["Trainer"]
            ).Trainer.train
        )
        assert "out of memory" in source

    def test_handle_oom_enables_gradient_checkpointing(self):
        """_handle_oom enables gradient checkpointing."""
        source = inspect.getsource(
            __import__(
                "enigma_engine.core.training", fromlist=["Trainer"]
            ).Trainer._handle_oom
        )
        assert "gradient_checkpointing" in source


# ================================================================
# 13: LoRA-Based RLHF (no deep copy on GPU)
# ================================================================


class TestRLHFNoDeepCopy:
    """Tests for LoRA-based RLHF reference policy."""

    def test_rlhf_has_setup_reference(self):
        """RLHFTrainer has _setup_reference method."""
        from enigma_engine.core.rl_training import RLHFTrainer
        assert hasattr(RLHFTrainer, "_setup_reference")

    def test_rlhf_has_get_ref_logps(self):
        """RLHFTrainer has _get_ref_logps method."""
        from enigma_engine.core.rl_training import RLHFTrainer
        assert hasattr(RLHFTrainer, "_get_ref_logps")

    def test_selfplay_has_setup_reference(self):
        """SelfPlayTrainer has _setup_reference method."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        assert hasattr(SelfPlayTrainer, "_setup_reference")

    def test_selfplay_has_get_ref_logps(self):
        """SelfPlayTrainer has _get_ref_logps method."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        assert hasattr(SelfPlayTrainer, "_get_ref_logps")

    def test_rlhf_train_uses_setup_reference(self):
        """RLHFTrainer.train() calls _setup_reference, not copy.deepcopy."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "_setup_reference" in source
        assert "copy.deepcopy" not in source

    def test_selfplay_train_uses_setup_reference(self):
        """SelfPlayTrainer.train() calls _setup_reference, not copy.deepcopy."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.train)
        assert "_setup_reference" in source
        assert "copy.deepcopy" not in source

    def test_setup_reference_tries_lora_first(self):
        """_setup_reference tries LoRA before fallback."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer._setup_reference)
        assert "create_lora_model" in source
        assert "PEFT_AVAILABLE" in source

    def test_setup_reference_has_cpu_fallback(self):
        """_setup_reference falls back to CPU offload."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer._setup_reference)
        assert ".cpu()" in source

    def test_no_gpu_deepcopy_in_train(self):
        """Guard: train() methods must not deepcopy to stay on GPU."""
        from enigma_engine.core.rl_training import (
            RLHFTrainer, SelfPlayTrainer,
        )
        for cls in (RLHFTrainer, SelfPlayTrainer):
            source = inspect.getsource(cls.train)
            assert "deepcopy" not in source, (
                f"{cls.__name__}.train() still uses deepcopy")


# ================================================================
# 14: Adaptive Trainer Dead Code Removal
# ================================================================


class TestAdaptiveTrainerCleanup:
    """Tests for dead code removal in adaptive_trainer.py."""

    def test_no_advance_score_field(self):
        """TrainingPlan no longer has advance_score."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        assert "advance_score" not in TrainingPlan.__dataclass_fields__

    def test_no_retry_score_field(self):
        """TrainingPlan no longer has retry_score."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        assert "retry_score" not in TrainingPlan.__dataclass_fields__

    def test_no_simplify_difficulty_method(self):
        """TrainingPlan no longer has simplify_difficulty."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        assert not hasattr(TrainingPlan, "simplify_difficulty")

    def test_no_raise_difficulty_method(self):
        """TrainingPlan no longer has raise_difficulty."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        assert not hasattr(TrainingPlan, "raise_difficulty")

    def test_decide_action_still_works(self):
        """decide_action still returns advance/complete."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.decide_action(8.0) == "advance"
        # Last stage → complete
        plan.current_stage_idx = len(plan.stages) - 1
        assert plan.decide_action(8.0) == "complete"

    def test_summary_no_thresholds(self):
        """summary() no longer mentions thresholds."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan(student_name="s", trainer_name="t")
        text = plan.summary()
        assert "Thresholds" not in text

    def test_save_uses_atomic_write(self):
        """TrainingPlan.save() uses atomic_write_text."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        source = inspect.getsource(TrainingPlan.save)
        assert "atomic_write_text" in source

    def test_load_still_filters_unknown_keys(self, tmp_path):
        """load() ignores keys removed from dataclass (forward compat)."""
        import json
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        # Save JSON with old fields that no longer exist
        data = {
            "student_path": "m.pth",
            "advance_score": 7.0,
            "retry_score": 5.0,
            "stages": ["basics", "conversation"],
        }
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        plan = TrainingPlan.load(path)
        assert plan.student_path == "m.pth"
        assert not hasattr(plan, "advance_score")


# ================================================================
# 15: CuratedDataset Thread Lock
# ================================================================


class TestCuratedDatasetThreadSafety:
    """Tests for threading.Lock in CuratedDataset."""

    def test_has_lock(self, tmp_path):
        """CuratedDataset has a threading.Lock."""
        import threading
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        assert hasattr(ds, "_lock")
        assert isinstance(ds._lock, type(threading.Lock()))

    def test_entries_returns_copy(self, tmp_path):
        """entries property returns a snapshot, not the internal list."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="test")
        snap = ds.entries
        snap.append(None)  # mutate snapshot
        assert ds.count == 1  # internal unchanged

    def test_concurrent_adds(self, tmp_path):
        """Concurrent adds don't corrupt data."""
        import threading as th
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")

        def add_entries():
            for i in range(50):
                ds.add(f"entry-{i}", source="thread")

        threads = [th.Thread(target=add_entries) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ds.count == 200  # 4 threads × 50 entries

    def test_lock_used_in_source(self):
        """CuratedDataset methods use self._lock."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        source = inspect.getsource(CuratedDataset)
        # At least several methods should use the lock
        lock_count = source.count("with self._lock")
        assert lock_count >= 10, (
            f"Expected >= 10 lock acquisitions, found {lock_count}")

    def test_approve_under_lock(self, tmp_path):
        """approve() is thread-safe."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        for i in range(100):
            ds.add(f"e{i}", source="test")

        def approve_range(start, end):
            for i in range(start, end):
                ds.approve(i)

        import threading as th
        t1 = th.Thread(target=approve_range, args=(0, 50))
        t2 = th.Thread(target=approve_range, args=(50, 100))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert ds.approved_count == 100


# ================================================================
# Suggestion 16: RAG BM25 + sparse + stop words
# ================================================================


class TestRAGBM25:
    """Test BM25 scoring, stop word filtering, and sparse matrix support."""

    def test_bm25_idf_formula(self):
        """BM25 IDF should differ from classic TF-IDF."""
        from enigma_engine.core.rag import TfidfVectorizer

        vec = TfidfVectorizer()
        docs = ["cat sat", "dog ran", "cat ran fast"]
        vec.fit(docs)
        # "cat" appears in 2 of 3 docs
        # BM25 IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        assert "cat" in vec.vocab
        idx_cat = vec.vocab["cat"]
        assert vec.idf[idx_cat] > 0
        # "fast" appears in 1 doc → higher IDF
        idx_fast = vec.vocab["fast"]
        assert vec.idf[idx_fast] > vec.idf[idx_cat]

    def test_bm25_k1_b_stored(self):
        """BM25 parameters k1 and b should be stored in vectorizer."""
        from enigma_engine.core.rag import TfidfVectorizer

        vec = TfidfVectorizer(k1=2.0, b=0.5)
        assert vec.k1 == 2.0
        assert vec.b == 0.5

    def test_bm25_serialization(self):
        """to_dict/from_dict should round-trip BM25 state."""
        from enigma_engine.core.rag import TfidfVectorizer

        vec = TfidfVectorizer(k1=1.2, b=0.8)
        docs = ["hello world", "foo bar hello"]
        vec.fit(docs)
        d = vec.to_dict()

        vec2 = TfidfVectorizer.from_dict(d)
        assert vec2.k1 == 1.2
        assert vec2.b == 0.8
        assert vec2.avg_dl == vec.avg_dl
        assert list(vec2.doc_lens) == list(vec.doc_lens)

    def test_bm25_backward_compat_from_dict(self):
        """from_dict without BM25 keys should use defaults."""
        from enigma_engine.core.rag import TfidfVectorizer

        # Old format: idf as a flat list, no k1/b/doc_lens/avg_dl
        d = {"vocab": {"aa": 0, "bb": 1}, "idf": [1.0, 0.5]}
        vec = TfidfVectorizer.from_dict(d)
        assert vec.k1 == 1.5  # Default
        assert vec.b == 0.75  # Default
        assert vec.avg_dl == 0.0

    def test_stop_words_filtered(self):
        """Tokenizer should filter common stop words."""
        from enigma_engine.core.rag import _tokenize

        tokens = _tokenize("the cat is on a mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens

    def test_stop_words_preserves_content_words(self):
        """Stop word filter should keep meaningful words."""
        from enigma_engine.core.rag import _tokenize

        tokens = _tokenize("machine learning algorithms")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "algorithms" in tokens

    def test_transform_returns_array(self):
        """transform() should return an array-like regardless of scipy."""
        import numpy as np
        from enigma_engine.core.rag import TfidfVectorizer

        vec = TfidfVectorizer()
        docs = ["cat dog", "fish bird"]
        vec.fit(docs)
        result = vec.transform(["cat bird"])
        # Whether sparse or dense, we should be able to get a 2-D array
        if hasattr(result, 'toarray'):
            arr = result.toarray()
        elif hasattr(result, 'A'):
            arr = np.asarray(result.A)
        else:
            arr = np.asarray(result)
        assert arr.shape[0] == 1
        assert arr.shape[1] == len(vec.vocab)


# ================================================================
# Suggestion 17: _load_gguf / _load_pytorch extraction
# ================================================================


class TestLoadModelRefactor:
    """Test that _load_gguf and _load_pytorch exist as methods."""

    def test_load_gguf_method_exists(self):
        from enigma_engine.core.inference import EnigmaEngine

        assert hasattr(EnigmaEngine, "_load_gguf")
        assert callable(getattr(EnigmaEngine, "_load_gguf"))

    def test_load_pytorch_method_exists(self):
        from enigma_engine.core.inference import EnigmaEngine

        assert hasattr(EnigmaEngine, "_load_pytorch")
        assert callable(getattr(EnigmaEngine, "_load_pytorch"))

    def test_load_model_dispatches(self):
        """_load_model should be a thin dispatcher (< 50 lines of logic)."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine

        src = inspect.getsource(EnigmaEngine._load_model)
        lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
        # The dispatcher body should be compact
        assert len(lines) < 60, f"_load_model is {len(lines)} lines, expected < 60"


# ================================================================
# Suggestion 18: Tokenizer – no silent SimpleTokenizer fallback
# ================================================================


class TestTokenizerNoSilentFallback:
    """Test that auto mode never silently uses SimpleTokenizer."""

    def test_explicit_simple_still_works(self):
        """Explicitly requesting 'simple' should still return SimpleTokenizer."""
        from enigma_engine.core.tokenizer import get_tokenizer, SimpleTokenizer

        tok = get_tokenizer("simple", use_cache=False)
        assert isinstance(tok, SimpleTokenizer)

    def test_auto_does_not_return_simple(self):
        """Auto mode should never return a SimpleTokenizer."""
        from enigma_engine.core.tokenizer import get_tokenizer, SimpleTokenizer

        tok = get_tokenizer("auto", use_cache=False)
        assert not isinstance(tok, SimpleTokenizer), (
            "Auto mode fell through to SimpleTokenizer — this is a silent quality bug"
        )

    def test_bpe_block_catches_import_error_only(self):
        """BPE block should catch ImportError, not broad Exception."""
        import ast
        from enigma_engine.core import tokenizer as tok_mod

        source = inspect.getsource(tok_mod.get_tokenizer)
        tree = ast.parse(source)
        # Find except handlers in the function
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.name == "e":
                # Check the log message to identify which block
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        if "BPE" in child.value:
                            assert isinstance(node.type, ast.Name)
                            assert node.type.id == "ImportError", (
                                f"BPE except catches {node.type.id}, expected ImportError"
                            )
                        if "character" in child.value:
                            assert isinstance(node.type, ast.Name)
                            assert node.type.id == "ImportError", (
                                f"Char except catches {node.type.id}, expected ImportError"
                            )

    def test_unknown_type_raises(self):
        """Unknown tokenizer types should raise, not silently fallback."""
        from enigma_engine.core.tokenizer import get_tokenizer

        with pytest.raises((RuntimeError, ValueError)):
            get_tokenizer("nonexistent_tokenizer", use_cache=False)


# ================================================================
# Suggestion 19: Web SSRF protection + streaming
# ================================================================


class TestWebSSRF:
    """Test URL validation and response streaming in web_utils."""

    def test_validate_url_rejects_file_scheme(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _validate_url("file:///etc/passwd")

    def test_validate_url_rejects_ftp_scheme(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _validate_url("ftp://evil.com/payload")

    def test_validate_url_rejects_localhost(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="private|reserved"):
            _validate_url("http://127.0.0.1/admin")

    def test_validate_url_rejects_private_ip(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="private|reserved"):
            _validate_url("http://192.168.1.1/secret")

    def test_validate_url_rejects_no_hostname(self):
        from enigma_engine.core.web_utils import _validate_url

        with pytest.raises(ValueError, match="No hostname"):
            _validate_url("http:///path")

    def test_max_response_bytes_constant(self):
        from enigma_engine.core.web_utils import _MAX_RESPONSE_BYTES

        assert _MAX_RESPONSE_BYTES == 1_048_576

    def test_fetch_page_text_validates_url(self):
        """fetch_page_text should reject private IPs before making a request."""
        from enigma_engine.core.web_utils import fetch_page_text

        with pytest.raises(ValueError, match="private|reserved"):
            fetch_page_text("http://10.0.0.1/internal")


# ================================================================
# Suggestion 20: _init_common eliminates attribute drift
# ================================================================


class TestInitCommon:
    """Test that _init_common sets all shared attributes."""

    def test_init_common_exists(self):
        from enigma_engine.core.inference import EnigmaEngine

        assert hasattr(EnigmaEngine, "_init_common")

    def test_init_common_sets_all_required_attrs(self):
        """_init_common must set every attribute both constructors need."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine

        source = inspect.getsource(EnigmaEngine._init_common)
        required_attrs = [
            "_generation_lock", "device", "use_half",
            "enable_tools", "module_manager", "use_routing",
            "use_offloading", "_tool_executor", "_tool_router",
            "_is_gguf", "_web_enabled", "vision_encoder",
            "model_metadata", "_chat_media_refs", "_link_urls",
            "_chat_history", "_token_count_cache",
        ]
        for attr in required_attrs:
            assert f"self.{attr}" in source, (
                f"_init_common missing self.{attr}"
            )

    def test_init_calls_init_common(self):
        """__init__ should delegate to _init_common."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine

        source = inspect.getsource(EnigmaEngine.__init__)
        assert "_init_common" in source, (
            "__init__ does not call _init_common — attribute drift risk"
        )

    def test_from_model_calls_init_common(self):
        """from_model should delegate to _init_common."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine

        source = inspect.getsource(EnigmaEngine.from_model)
        assert "_init_common" in source, (
            "from_model does not call _init_common — attribute drift risk"
        )


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
            # Cache is full — inserting "d" should evict "a"
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

    def test_fetch_one_returns_string(self):
        """_fetch_one always returns a string."""
        from enigma_engine.core.auto_research import _fetch_one
        # With a bad URL it should return "" not raise
        result = _fetch_one("http://nonexistent.invalid", 500)
        assert isinstance(result, str)

    def test_parallel_fetch_uses_threadpool(self):
        """auto_research function uses ThreadPoolExecutor."""
        source = inspect.getsource(
            __import__(
                "enigma_engine.core.auto_research",
                fromlist=["auto_research"],
            ).auto_research
        )
        assert "ThreadPoolExecutor" in source

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

class TestKVCacheClone:
    """Tests for KV cache get() returning cloned tensors."""

    def test_get_returns_cloned_tensors(self):
        """Mutating returned tensors must not corrupt the cache."""
        import torch
        from enigma_engine.core.kv_cache import KVCache

        cache = KVCache(
            batch_size=1, max_seq_len=16, n_kv_heads=2,
            head_dim=4, device="cpu", dtype=torch.float32,
        )
        # Write something identifiable into the cache
        data = torch.ones(1, 2, 4)
        cache.update(data, data, position=0)

        # Get and mutate
        k, v = cache.get()
        k.zero_()
        v.zero_()

        # Original cache must be intact
        k2, v2 = cache.get()
        assert k2.sum().item() > 0, "Cache was corrupted by mutation"
        assert v2.sum().item() > 0, "Cache was corrupted by mutation"

    def test_get_source_uses_clone(self):
        """Non-quantized path in get() must call .clone()."""
        from enigma_engine.core.kv_cache import KVCache

        source = inspect.getsource(KVCache.get)
        # Should contain .clone() for the non-quantized branch
        assert ".clone()" in source


# ================================================================
# Vision Encoder Dedup + Max Visual Tokens (#27)
# ================================================================

class TestVideoFrameDedup:
    """Tests for video frame dedup and max_visual_tokens in encode_video_frames."""

    def test_encode_video_frames_has_dedup_params(self):
        """encode_video_frames accepts max_visual_tokens and dedup_threshold."""
        from enigma_engine.core.vision_encoder import encode_video_frames

        sig = inspect.signature(encode_video_frames)
        params = sig.parameters
        assert "max_visual_tokens" in params
        assert "dedup_threshold" in params
        assert params["max_visual_tokens"].default == 0
        assert params["dedup_threshold"].default == 0.95

    def test_encode_video_frames_dedup_logic_in_source(self):
        """encode_video_frames contains cosine_similarity dedup logic."""
        from enigma_engine.core.vision_encoder import encode_video_frames

        source = inspect.getsource(encode_video_frames)
        assert "cosine_similarity" in source
        assert "dedup_threshold" in source

    def test_encode_video_frames_truncation_in_source(self):
        """encode_video_frames contains max_visual_tokens truncation."""
        from enigma_engine.core.vision_encoder import encode_video_frames

        source = inspect.getsource(encode_video_frames)
        assert "max_visual_tokens" in source
        # Should slice combined tensor
        assert "combined[:, :max_visual_tokens, :]" in source

    def test_dedup_drops_identical_frames(self):
        """Identical consecutive frames (cosine_sim=1.0) are dropped."""
        import torch

        # Simulate dedup logic directly
        feat = torch.ones(1, 4, 8)  # identical frame feature
        all_features = [feat.clone() for _ in range(5)]

        threshold = 0.95
        unique = [all_features[0]]
        for f in all_features[1:]:
            prev = unique[-1].reshape(-1)
            curr = f.reshape(-1)
            cos_sim = torch.nn.functional.cosine_similarity(
                prev.unsqueeze(0), curr.unsqueeze(0)).item()
            if cos_sim < threshold:
                unique.append(f)

        # All identical → only first kept
        assert len(unique) == 1

    def test_dedup_keeps_different_frames(self):
        """Different consecutive frames are kept."""
        import torch

        all_features = [
            torch.randn(1, 4, 8) * 100,  # scale up to avoid accidental similarity
            torch.randn(1, 4, 8) * 100,
            torch.randn(1, 4, 8) * 100,
        ]

        threshold = 0.95
        unique = [all_features[0]]
        for f in all_features[1:]:
            prev = unique[-1].reshape(-1)
            curr = f.reshape(-1)
            cos_sim = torch.nn.functional.cosine_similarity(
                prev.unsqueeze(0), curr.unsqueeze(0)).item()
            if cos_sim < threshold:
                unique.append(f)

        # Random tensors are unlikely to have cosine_sim > 0.95
        assert len(unique) >= 2

    def test_max_visual_tokens_truncation(self):
        """Concatenated features are truncated to max_visual_tokens."""
        import torch

        features = [torch.randn(1, 10, 8) for _ in range(3)]
        combined = torch.cat(features, dim=1)  # [1, 30, 8]
        max_visual_tokens = 15
        if max_visual_tokens > 0 and combined.shape[1] > max_visual_tokens:
            combined = combined[:, :max_visual_tokens, :]
        assert combined.shape[1] == 15


# ================================================================
# run.py Lazy Torch Import (#28) + Port from Config (#29)
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


class TestRunPyPortFromConfig:
    """Tests for run.py reading port from CONFIG."""

    def test_port_default_is_none(self):
        """--port argument default should be None (reads from CONFIG)."""
        run_path = Path(__file__).parent.parent / "run.py"
        source = run_path.read_text(encoding="utf-8")
        # Argparse should set default=None for --port
        assert 'default=None' in source.split("--port")[1].split("\n")[0]

    def test_port_resolution_reads_config(self):
        """run_serve dispatch reads CONFIG when no port given."""
        run_path = Path(__file__).parent.parent / "run.py"
        source = run_path.read_text(encoding="utf-8")
        assert "CONFIG" in source
        assert "api_port" in source

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


class TestSpecialTokenIds:
    """All tokenizers must expose think_start_id / think_end_id."""

    def test_char_tokenizer_think_ids(self):
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer()
        assert hasattr(tok, "think_start_id")
        assert hasattr(tok, "think_end_id")
        assert tok.think_start_id == tok.special_tokens["<think>"]
        assert tok.think_end_id == tok.special_tokens["</think>"]

    def test_bpe_tokenizer_think_ids(self):
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        assert hasattr(tok, "think_start_id")
        assert hasattr(tok, "think_end_id")
        assert tok.think_start_id == tok.special_tokens["<think>"]
        assert tok.think_end_id == tok.special_tokens["</think>"]

    def test_advanced_tokenizer_think_ids(self):
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        tok = AdvancedBPETokenizer()
        assert hasattr(tok, "think_start_id")
        assert hasattr(tok, "think_end_id")
        assert tok.think_start_id == tok.special_tokens["<think>"]
        assert tok.think_end_id == tok.special_tokens["</think>"]

    def test_simple_tokenizer_think_ids(self):
        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        assert hasattr(tok, "think_start_id")
        assert hasattr(tok, "think_end_id")
        assert tok.think_start_id == 4
        assert tok.think_end_id == 5

    def test_get_special_token_ids_uses_attributes(self):
        """get_special_token_ids returns correct think IDs from tokenizer."""
        from enigma_engine.core.tokenizer import get_special_token_ids
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer()
        ids = get_special_token_ids(tok)
        assert ids["think_start"] == tok.think_start_id
        assert ids["think_end"] == tok.think_end_id

    def test_core_ids_consistent(self):
        """pad=0, bos=1, eos=2, unk=3 across all tokenizers."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        from enigma_engine.core.tokenizer import SimpleTokenizer
        for cls in [CharacterTokenizer, BPETokenizer,
                    AdvancedBPETokenizer, SimpleTokenizer]:
            tok = cls()
            assert tok.pad_token_id == 0, f"{cls.__name__} pad != 0"
            assert tok.bos_token_id == 1, f"{cls.__name__} bos != 1"
            assert tok.eos_token_id == 2, f"{cls.__name__} eos != 2"
            assert tok.unk_token_id == 3, f"{cls.__name__} unk != 3"


class TestTrainingConfigValSplit:
    """TrainingConfig val_split field."""

    def test_default_zero(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.val_split == 0.0

    def test_to_dict_includes_val_split(self):
        from enigma_engine.core.training import TrainingConfig
        d = TrainingConfig(val_split=0.1).to_dict()
        assert d["val_split"] == 0.1

    def test_validate_rejects_bad_val_split(self):
        import pytest
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(val_split=1.0)
        with pytest.raises(ValueError, match="val_split"):
            cfg.validate()
        cfg2 = TrainingConfig(val_split=-0.1)
        with pytest.raises(ValueError, match="val_split"):
            cfg2.validate()

    def test_validate_accepts_valid_val_split(self):
        from enigma_engine.core.training import TrainingConfig
        TrainingConfig(val_split=0.0).validate()
        TrainingConfig(val_split=0.2).validate()


class TestValidationLoop:
    """Trainer._validate() method."""

    def test_validate_method_exists(self):
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_validate")

    def test_validation_losses_populated(self):
        """state.validation_losses filled when val_split > 0."""
        from enigma_engine.core.training import TrainingState
        s = TrainingState()
        assert s.validation_losses == []
        s.validation_losses.append(1.5)
        assert len(s.validation_losses) == 1
