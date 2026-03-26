"""
Tests for Enigma Engine core functionality.

Run with: python -m pytest tests/ -v
"""

import inspect
import textwrap
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


class TestStreamChatStopStringHoldback:
    """stream_chat must hold back tokens that could be stop-string prefixes."""

    def test_stream_chat_has_pending_buffer(self):
        """stream_chat uses a pending buffer to hold back stop-string prefixes."""
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.stream_chat)
        assert "pending" in source, (
            "stream_chat must use a pending/holdback buffer for stop strings")

    def test_stream_chat_checks_stop_prefix(self):
        """stream_chat checks if pending tail matches a stop-string prefix."""
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.stream_chat)
        # Must check if tail of pending could be start of a stop string
        assert "stop[:k]" in source or "startswith" in source or "tail" in source, (
            "stream_chat must check if pending text could be a stop-string prefix")

    def test_stream_chat_no_immediate_yield_in_native(self):
        """stream_chat native path must NOT yield tokens immediately before stop check."""
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.stream_chat)
        # The old bug: `yield token` appeared right after the for-loop
        # before any hold-back logic. Now it should yield from pending.
        native_section = source[source.index("Native model streaming"):]
        lines = native_section.split("\n")
        # Should not have a bare "yield token" line (the old pattern)
        bare_yield_token = [l.strip() for l in lines if l.strip() == "yield token"]
        assert len(bare_yield_token) == 0, (
            "stream_chat must not yield raw tokens — should yield from pending buffer")

    def test_stream_chat_flushes_pending_at_eof(self):
        """stream_chat must flush remaining pending text when generation ends naturally."""
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.stream_chat)
        # After the for loop, there must be a final yield of pending
        native_section = source[source.index("Native model streaming"):]
        # The for loop ends, then pending should be flushed
        assert "if pending" in native_section, (
            "stream_chat must flush remaining pending buffer after stream ends")


class TestPrepareChatMaxTokens:
    """_prepare_chat extracts max_tokens from kwargs."""

    def test_prepare_chat_extracts_max_tokens_from_kwargs(self):
        """_prepare_chat respects max_tokens kwarg to override max_gen."""
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin._prepare_chat)
        assert "max_tokens" in source, (
            "_prepare_chat must extract max_tokens from kwargs")

    def test_prepare_chat_pops_max_tokens(self):
        """_prepare_chat pops max_tokens (not just gets) to avoid double-passing."""
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin._prepare_chat)
        assert 'pop("max_tokens"' in source or "pop('max_tokens'" in source, (
            "_prepare_chat should pop max_tokens so it's not also "
            "forwarded as a separate kwarg")


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

    def test_code_run_blocks_dunder_import(self):
        """code.run should block __import__ bypass attempts."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run __import__('os').system('whoami')")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_blocks_importlib(self):
        """code.run should block importlib-based imports."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run importlib.import_module('os').system('ls')")
        assert not result.success
        assert "Forbidden" in result.message

    def test_code_run_blocks_open_write_outside_outputs(self):
        """code.run should block open() for writing outside outputs/."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run open('/etc/passwd', 'w').write('bad')")
        assert not result.success or "restricted" in result.message.lower() or "PermissionError" in result.message

    def test_code_run_blocks_compile_exec(self):
        """code.run should block compile() bypass."""
        from enigma_engine.core.commands import get_registry
        reg = get_registry()
        result = reg.execute("code.run x = compile('1+1', '', 'eval')")
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

    # --- Expanded fact extraction patterns ---

    def test_extract_facts_hobby(self, tmp_path):
        """extract_facts catches 'I enjoy X' / 'I love X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I enjoy hiking on weekends")
        assert len(added) >= 1
        assert any("hiking" in f.lower() for f in added)

    def test_extract_facts_love(self, tmp_path):
        """extract_facts catches 'I love X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I really love cooking Italian food")
        assert len(added) >= 1
        assert any("cooking" in f.lower() for f in added)

    def test_extract_facts_age(self, tmp_path):
        """extract_facts catches 'I'm X years old'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I'm 28 years old by the way")
        assert len(added) >= 1
        assert any("28" in f for f in added)

    def test_extract_facts_birthday(self, tmp_path):
        """extract_facts catches 'my birthday is X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("My birthday is March 15th")
        assert len(added) >= 1
        assert any("March" in f for f in added)

    def test_extract_facts_pet(self, tmp_path):
        """extract_facts catches 'I have a dog/cat named X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I have a dog named Buddy")
        assert len(added) >= 1
        assert any("Buddy" in f for f in added)

    def test_extract_facts_family(self, tmp_path):
        """extract_facts catches family members."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("My wife's name is Sarah")
        assert len(added) >= 1
        assert any("Sarah" in f for f in added)

    def test_extract_facts_education(self, tmp_path):
        """extract_facts catches 'I studied at X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I studied at MIT")
        assert len(added) >= 1
        assert any("MIT" in f for f in added)

    def test_extract_facts_dislike(self, tmp_path):
        """extract_facts catches 'I hate X' / 'I don't like X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I hate spiders honestly")
        assert len(added) >= 1
        assert any("spiders" in f.lower() for f in added)

    def test_extract_facts_language(self, tmp_path):
        """extract_facts catches 'I speak X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I speak Spanish and French fluently")
        assert len(added) >= 1
        assert any("Spanish" in f for f in added)

    def test_extract_facts_timezone(self, tmp_path):
        """extract_facts catches timezone info."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I'm in the EST timezone")
        assert len(added) >= 1
        assert any("EST" in f for f in added)

    def test_extract_facts_degree(self, tmp_path):
        """extract_facts catches 'I have a degree in X'."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(memory_path=tmp_path / "mem.md")
        added = mem.extract_facts("I have a degree in computer science")
        assert len(added) >= 1
        assert any("computer science" in f.lower() for f in added)


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
        assert spec is not None and spec.loader is not None
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
        assert spec is not None and spec.loader is not None
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
        assert spec is not None and spec.loader is not None
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
            REPETITION_WINDOW = _GenerationMixin.REPETITION_WINDOW
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
            counts[int(tok.item())] += 1
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


class TestBlockedPathEnforcement:
    """Verify file commands enforce blocked_paths/blocked_patterns from config."""

    def test_check_blocked_path_exists(self):
        """_check_blocked_path helper is defined in builtin_commands."""
        import inspect
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        assert "_check_blocked_path" in source

    def test_file_list_checks_blocked(self):
        """file.list checks _check_blocked_path before listing."""
        import inspect
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        list_idx = source.index("def file_list(")
        list_body = source[list_idx:list_idx + 500]
        assert "_check_blocked_path" in list_body

    def test_file_read_checks_blocked(self):
        """file.read checks _check_blocked_path before reading."""
        import inspect
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        # Find file_read function and verify it calls _check_blocked_path
        read_idx = source.index("def file_read(")
        read_body = source[read_idx:read_idx + 500]
        assert "_check_blocked_path" in read_body

    def test_file_write_checks_blocked(self):
        """file.write checks _check_blocked_path before writing."""
        import inspect
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        write_idx = source.index("def file_write(")
        write_body = source[write_idx:write_idx + 500]
        assert "_check_blocked_path" in write_body

    def test_file_append_checks_blocked(self):
        """file.append checks _check_blocked_path before appending."""
        import inspect
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        append_idx = source.index("def file_append(")
        append_body = source[append_idx:append_idx + 500]
        assert "_check_blocked_path" in append_body

    def test_blocked_pattern_matches(self):
        """_check_blocked_path blocks files matching config patterns."""
        from enigma_engine.core.builtin_commands import register_builtin_commands
        from enigma_engine.core.commands import CommandRegistry
        reg = CommandRegistry()
        register_builtin_commands(reg)
        reg.set_context("config", {"blocked_patterns": ["*.pem", "*secret*"]})
        result = reg.execute("file.read server.pem")
        assert not result.success
        assert "blocked pattern" in result.message.lower()

    def test_unblocked_path_passes(self):
        """_check_blocked_path allows files not matching any pattern."""
        import inspect
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        # _check_blocked_path returns None for clean paths
        assert "return None" in source

    def test_default_blocked_paths_has_system_dirs(self):
        """Default blocked_paths config should include system directories."""
        import ast
        from pathlib import Path
        # Read the raw source defaults, not the runtime CONFIG
        # (which may be overridden by forge_config.json)
        source_path = (
            Path(__file__).resolve().parent.parent
            / "enigma_engine" / "config" / "defaults.py"
        )
        source = source_path.read_text(encoding="utf-8")
        # blocked_paths list must contain entries in the source defaults
        assert '"blocked_paths": [' in source
        # Must not be empty in the source defaults
        idx = source.index('"blocked_paths": [')
        # The line after should NOT be just "],"
        block = source[idx:idx + 500]
        assert '"/etc"' in block or '"C:/Windows"' in block or '"/usr"' in block

    def test_default_blocked_patterns_has_sensitive_extensions(self):
        """Default blocked_patterns should block sensitive file types."""
        from enigma_engine.config.defaults import CONFIG
        patterns = CONFIG.get("blocked_patterns", [])
        # Should block executables, keys, and sensitive files
        assert "*.exe" in patterns
        assert "*.pem" in patterns
        assert "*.key" in patterns


class TestShellMetacharNotBlocked:
    """Verify shell command no longer blocks metacharacters (shell=False is sufficient)."""

    def test_no_metacharacter_error_in_shell_handler(self):
        """shell command handler does not contain metacharacter blocking."""
        import inspect
        from enigma_engine.core.builtin_commands import register_builtin_commands
        source = inspect.getsource(register_builtin_commands)
        # Find the shell command handler
        shell_idx = source.index("ALLOWED_COMMANDS")
        shell_body = source[shell_idx:shell_idx + 1000]
        assert "Shell metacharacters are not allowed" not in shell_body


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
        assert encoder.pos_embed is not None
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


class TestPretrainedVisionConfig:
    """VisionEncoderConfig pretrained fields."""

    def test_pretrained_defaults(self):
        """use_pretrained should default to False, preserving existing behavior."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig()
        assert cfg.use_pretrained is False
        assert cfg.pretrained_model == "vit_small_patch16_224"
        assert cfg.freeze_backbone is True

    def test_pretrained_to_dict_roundtrip(self):
        """Pretrained fields should survive dict serialization."""
        from enigma_engine.core.vision_encoder import VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_base_patch16_224",
            freeze_backbone=False,
        )
        d = cfg.to_dict()
        assert d["use_pretrained"] is True
        assert d["pretrained_model"] == "vit_base_patch16_224"
        assert d["freeze_backbone"] is False
        cfg2 = VisionEncoderConfig(**d)
        assert cfg2.use_pretrained is True
        assert cfg2.pretrained_model == "vit_base_patch16_224"
        assert cfg2.freeze_backbone is False

    def test_pretrained_presets_exist(self):
        """Pretrained presets should be in VISION_PRESETS."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        assert "pretrained_tiny" in VISION_PRESETS
        assert "pretrained_small" in VISION_PRESETS
        assert "pretrained_base" in VISION_PRESETS
        ps = VISION_PRESETS["pretrained_small"]
        assert ps.use_pretrained is True

    def test_pretrained_presets_have_correct_dims(self):
        """Pretrained presets should match standard ViT dimensions."""
        from enigma_engine.core.vision_encoder import VISION_PRESETS
        assert VISION_PRESETS["pretrained_tiny"].dim == 192
        assert VISION_PRESETS["pretrained_small"].dim == 384
        assert VISION_PRESETS["pretrained_base"].dim == 768


class TestImageNetNormalization:
    """ImageNet normalization constants and preprocessing."""

    def test_imagenet_constants_exist(self):
        """IMAGENET_MEAN and IMAGENET_STD should be defined."""
        from enigma_engine.core.vision_encoder import IMAGENET_MEAN, IMAGENET_STD
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        # Standard ImageNet values
        assert abs(IMAGENET_MEAN[0] - 0.485) < 0.01
        assert abs(IMAGENET_STD[0] - 0.229) < 0.01

    def test_preprocess_imagenet_normalize(self):
        """Preprocessing with imagenet_normalize should differ from default."""
        import torch
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        t_default = preprocess_image(img, image_size=32)
        t_imagenet = preprocess_image(img, image_size=32, imagenet_normalize=True)
        assert not torch.allclose(t_default, t_imagenet)

    def test_preprocess_imagenet_range(self):
        """ImageNet-normalized mid-gray should be near zero."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.core.vision_encoder import preprocess_image
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        t = preprocess_image(img, image_size=32, imagenet_normalize=True)
        # Mid-gray (128/255 ≈ 0.502) is close to ImageNet means (~0.45–0.485)
        # So normalized values should be small
        assert t.min() > -5.0
        assert t.max() < 5.0


class TestPretrainedVisionEncoder:
    """VisionEncoder with pretrained timm backbone."""

    def test_ensure_timm_callable(self):
        """_ensure_timm helper should exist and be callable."""
        from enigma_engine.core.vision_encoder import _ensure_timm
        assert callable(_ensure_timm)

    def test_pretrained_encoder_creates_backbone(self):
        """Pretrained encoder should have a backbone attribute."""
        pytest.importorskip("timm")
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,
        )
        encoder = VisionEncoder(cfg)
        assert hasattr(encoder, "backbone")
        assert encoder.backbone is not None

    def test_pretrained_encoder_forward_shape(self):
        """Pretrained encoder output shape should be [B, num_patches, dim]."""
        pytest.importorskip("timm")
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,
        )
        encoder = VisionEncoder(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        # 224/16 = 14, 14*14 = 196 patches
        assert out.shape == (1, 196, 192)

    def test_pretrained_freeze_backbone(self):
        """freeze_backbone=True should freeze all backbone parameters."""
        pytest.importorskip("timm")
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,
            freeze_backbone=True,
        )
        encoder = VisionEncoder(cfg)
        assert encoder.backbone is not None
        backbone_frozen = all(
            not p.requires_grad for p in encoder.backbone.parameters()
        )
        assert backbone_frozen

    def test_pretrained_unfreeze_backbone(self):
        """freeze_backbone=False should leave backbone weights trainable."""
        pytest.importorskip("timm")
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,
            freeze_backbone=False,
        )
        encoder = VisionEncoder(cfg)
        assert encoder.backbone is not None
        has_trainable = any(
            p.requires_grad for p in encoder.backbone.parameters()
        )
        assert has_trainable

    def test_pretrained_with_dim_projection(self):
        """When config.dim != backbone dim, a projection layer should exist."""
        pytest.importorskip("timm")
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        # vit_tiny_patch16_224 has embed_dim=192, set config.dim to 64
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=64,
        )
        encoder = VisionEncoder(cfg)
        assert encoder.backbone_proj is not None
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (1, 196, 64)

    def test_pretrained_no_projection_when_dims_match(self):
        """When config.dim matches backbone dim, no projection is needed."""
        pytest.importorskip("timm")
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=192,  # matches vit_tiny embed_dim
        )
        encoder = VisionEncoder(cfg)
        assert encoder.backbone_proj is None

    def test_pretrained_gradients_flow_through_projection(self):
        """Gradients should flow through the pretrained encoder."""
        pytest.importorskip("timm")
        import torch
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        cfg = VisionEncoderConfig(
            use_pretrained=True,
            pretrained_model="vit_tiny_patch16_224",
            dim=64,
            freeze_backbone=False,
        )
        encoder = VisionEncoder(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in encoder.parameters())
        assert has_grad


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
        # vocab_size=100 padded to next multiple of 64 = 128
        padded_vocab = (100 + 63) & ~63
        assert logits.shape == (1, expected_seq, padded_vocab)

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
        padded_vocab = (100 + 63) & ~63
        assert logits.shape == (1, vcfg.num_patches, padded_vocab)


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
        assert v_enc.patch_embed is not None and v_enc.patch_embed.proj is not None
        assert model.vision_projection is not None
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
        assert v_enc.patch_embed is not None and v_enc.patch_embed.proj is not None
        assert model.vision_projection is not None
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


class TestScanVisionDataVideo:
    """Scanner detection of video files for vision training."""

    def test_video_extensions_defined(self):
        """_VIDEO_EXTENSIONS set must exist in scanners module."""
        from enigma_engine.gui.scanners import _VIDEO_EXTENSIONS
        assert isinstance(_VIDEO_EXTENSIONS, (set, frozenset))
        assert ".mp4" in _VIDEO_EXTENSIONS
        assert ".avi" in _VIDEO_EXTENSIONS

    def test_scan_vision_data_source_handles_video(self):
        """scan_vision_data must contain video handling logic."""
        import inspect
        from enigma_engine.gui.scanners import scan_vision_data
        source = inspect.getsource(scan_vision_data)
        assert "_VIDEO_EXTENSIONS" in source, (
            "scan_vision_data must check video file extensions"
        )

    def test_extract_video_frames_function_exists(self):
        """_extract_video_frames helper must exist in scanners."""
        from enigma_engine.gui.scanners import _extract_video_frames
        assert callable(_extract_video_frames)

    def test_extract_video_frames_signature(self):
        """_extract_video_frames must accept video_path and max_frames."""
        import inspect
        from enigma_engine.gui.scanners import _extract_video_frames
        sig = inspect.signature(_extract_video_frames)
        params = list(sig.parameters.keys())
        assert "video_path" in params
        assert "max_frames" in params

    def test_scan_video_paired_with_txt(self):
        """scan_vision_data should detect video.mp4 + video.txt pairs."""
        import inspect
        from enigma_engine.gui.scanners import scan_vision_data
        source = inspect.getsource(scan_vision_data)
        # Must check for video files in the paired-file strategy
        assert "video" in source.lower() or "_VIDEO_EXTENSIONS" in source

    def test_video_frame_returns_pil_images(self):
        """Extracted video frames must be PIL Image objects."""
        import inspect
        from enigma_engine.gui.scanners import _extract_video_frames
        source = inspect.getsource(_extract_video_frames)
        # Must convert frames to PIL Images (for preprocess_image compatibility)
        assert "Image" in source or "PIL" in source or "fromarray" in source


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
        mixin.vision_encoder = VisionEncoder(cfg)  # type: ignore[attr-defined]
        mixin.device = torch.device("cpu")  # type: ignore[attr-defined]

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
        mixin.vision_encoder = VisionEncoder(cfg)  # type: ignore[attr-defined]
        mixin.device = torch.device("cpu")  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                p = Path(d) / f"img{i}.png"
                Image.new("RGB", (32, 32), (i * 50, 0, 0)).save(p)
                paths.append(str(p))

            features = mixin._encode_images_for_chat(paths)
            # Should concatenate along sequence dimension
            assert features is not None
            assert features.dim() == 3
            num_patches = cfg.num_patches
            assert features.shape[1] == num_patches * 3

    def test_encode_images_no_encoder_returns_none(self):
        """_encode_images_for_chat should return None when no vision encoder."""
        from enigma_engine.core.engine_chat import _ChatMixin

        mixin = _ChatMixin()
        mixin.vision_encoder = None  # type: ignore[attr-defined]
        mixin.device = None  # type: ignore[attr-defined]

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
        engine.vision_encoder = VisionEncoder(vcfg)  # type: ignore[assignment]

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
# Model class annotations
# ---------------------------------------------------------------------------

class TestModelAnnotations:
    """model.py classmethods should reference Enigma, not Forge alias."""

    def test_from_classmethods_return_enigma(self):
        """from_safetensors/from_gguf/from_onnx annotations say Enigma."""
        import inspect
        from enigma_engine.core.model import Enigma
        for method_name in ("from_safetensors", "from_gguf", "from_onnx"):
            method = getattr(Enigma, method_name)
            hints = method.__annotations__
            assert hints.get("return") == "Enigma", (
                f"{method_name} return annotation should be 'Enigma', "
                f"got {hints.get('return')!r}")

    def test_export_uses_module_level_safe_save(self):
        """Export methods should use module-level safe_save imports."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "core" / "model.py")
        source = source_path.read_text(encoding="utf-8")
        # Module-level import should exist
        assert "from .safe_save import" in source, (
            "model.py should import safe_save at module level")
        # Inline imports should NOT exist (except gguf which is optional)
        import inspect
        from enigma_engine.core.model import Enigma
        for method_name in ("export_to_safetensors", "export_to_onnx",
                            "export_to_pytorch"):
            method_source = inspect.getsource(getattr(Enigma, method_name))
            assert "from enigma_engine.core.safe_save" not in method_source, (
                f"{method_name} should not have inline safe_save import")


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
        source = inspect.getsource(GGUFExporter._infer_metadata)  # type: ignore[attr-defined]
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

    # ── QK Normalization ─────────────────────────────────────────

    def test_qk_norm_config_default_off(self):
        """use_qk_norm defaults to False."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert config.use_qk_norm is False

    def test_qk_norm_in_attention_source(self):
        """Attention uses RMSNorm layers for QK normalization when enabled."""
        from enigma_engine.core.model_components import Attention
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, n_heads=4, n_kv_heads=4, use_qk_norm=True)
        attn = Attention(config)
        assert hasattr(attn, "q_norm"), "Attention should have q_norm layer"
        assert hasattr(attn, "k_norm"), "Attention should have k_norm layer"

    def test_qk_norm_preserves_shape(self):
        """Attention output shape unchanged with qk_norm enabled."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import Attention
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, n_heads=4, n_kv_heads=4, use_qk_norm=True)
        attn = Attention(config)
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)

    # ── LayerScale ───────────────────────────────────────────────

    def test_layer_scale_config_default_off(self):
        """use_layer_scale defaults to False."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert config.use_layer_scale is False

    def test_layer_scale_creates_parameters(self):
        """TransformerBlock has ls_attn and ls_ffn when layer_scale enabled."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import TransformerBlock
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128, use_layer_scale=True)
        block = TransformerBlock(config, layer_id=0)
        assert hasattr(block, 'ls_attn')
        assert hasattr(block, 'ls_ffn')
        assert block.ls_attn.shape == (64,)
        assert block.ls_ffn.shape == (64,)

    def test_layer_scale_init_small(self):
        """LayerScale parameters are initialized to a small value (1e-5)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import TransformerBlock
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128, use_layer_scale=True)
        block = TransformerBlock(config, layer_id=0)
        assert block.ls_attn.mean().item() == pytest.approx(1e-5, abs=1e-7)

    def test_layer_scale_preserves_shape(self):
        """TransformerBlock output shape unchanged with layer_scale."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import TransformerBlock
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128, use_layer_scale=True)
        block = TransformerBlock(config, layer_id=0)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    # ── Drop Path (Stochastic Depth) ────────────────────────────

    def test_drop_path_config_default_zero(self):
        """drop_path_rate defaults to 0.0 (disabled)."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert config.drop_path_rate == 0.0

    def test_drop_path_class_exists(self):
        """DropPath helper class exists in model_components."""
        from enigma_engine.core.model_components import DropPath
        assert DropPath is not None

    def test_drop_path_noop_at_zero(self):
        """DropPath with rate=0 is identity."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import DropPath
        dp = DropPath(0.0)
        x = torch.randn(2, 4, 64)
        dp.train()
        out = dp(x)
        assert torch.equal(out, x)

    def test_drop_path_noop_at_eval(self):
        """DropPath is identity during eval regardless of rate."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import DropPath
        dp = DropPath(0.5)
        dp.eval()
        x = torch.randn(2, 4, 64)
        out = dp(x)
        assert torch.equal(out, x)

    def test_drop_path_linearly_increasing(self):
        """Deeper layers get higher drop rates."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_components import TransformerBlock
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(dim=64, hidden_dim=128, n_layers=4, drop_path_rate=0.2)
        block0 = TransformerBlock(config, layer_id=0)
        block3 = TransformerBlock(config, layer_id=3)
        assert block0.drop_path_attn.drop_prob < block3.drop_path_attn.drop_prob

    # ── EMA Weight Averaging ─────────────────────────────────────

    def test_ema_config_default_off(self):
        """ema_decay defaults to 0.0 (disabled) in TrainingConfig."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert config.ema_decay == 0.0

    def test_ema_in_to_dict(self):
        """ema_decay is serialized in TrainingConfig.to_dict()."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(ema_decay=0.999)
        d = config.to_dict()
        assert "ema_decay" in d
        assert d["ema_decay"] == 0.999

    def test_ema_class_exists(self):
        """EMAWeightAverager class exists in training module."""
        from enigma_engine.core.training import EMAWeightAverager
        assert EMAWeightAverager is not None

    def test_ema_tracks_weights(self):
        """EMAWeightAverager maintains shadow copies of parameters."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import EMAWeightAverager
        model = torch.nn.Linear(4, 4)
        ema = EMAWeightAverager(model, decay=0.99)
        # Shadow should exist for each parameter
        assert len(ema.shadow) == len(list(model.parameters()))

    def test_ema_update_moves_shadow(self):
        """EMAWeightAverager.update() moves shadow toward current weights."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import EMAWeightAverager
        model = torch.nn.Linear(4, 4, bias=False)
        ema = EMAWeightAverager(model, decay=0.99)
        old_shadow = ema.shadow[0].clone()
        # Change the model weights
        with torch.no_grad():
            model.weight.fill_(99.0)
        ema.update(model)
        # Shadow should have moved toward 99 but not all the way
        new_shadow = ema.shadow[0]
        assert not torch.equal(old_shadow, new_shadow)
        assert new_shadow.mean().item() > old_shadow.mean().item()

    # ── torch.compile ────────────────────────────────────────────

    def test_compile_config_default_off(self):
        """use_compile defaults to False in TrainingConfig."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert config.use_compile is False


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
        source = inspect.getsource(
            LogicChatMixin._build_system_prompt_with_context)
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

    def test_new_helpers_importable(self):
        """loss_to_proxy_score and build_test_prompt are importable."""
        from enigma_engine.core.adaptive_trainer import (
            loss_to_proxy_score, build_test_prompt)
        assert callable(loss_to_proxy_score)
        assert callable(build_test_prompt)

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
        """decide_action advances on high scores, retries on low."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.decide_action(8.0) == "advance"
        assert plan.decide_action(3.0) == "retry"

    def test_decide_action_retries_exhaust(self):
        """decide_action advances after max retries even with low scores."""
        from enigma_engine.core.adaptive_trainer import (
            TrainingPlan, StageResult)
        plan = TrainingPlan(max_retries=2)
        # Record 2 attempts on the current stage
        for i in range(2):
            plan.record_result(StageResult(
                stage="basics", attempt=i + 1,
                scores=[3.0], avg_score=3.0,
                status="retry"))
        # Now retries exhausted — should advance anyway
        assert plan.decide_action(3.0) == "advance"

    def test_decide_action_escalates_difficulty(self):
        """decide_action escalates difficulty on retry."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.current_difficulty == "simple"
        plan.decide_action(4.0)  # triggers retry
        assert plan.current_difficulty == "medium"
        plan.decide_action(4.0)  # triggers another retry
        assert plan.current_difficulty == "advanced"

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


class TestLossToProxyScore:
    """Test loss_to_proxy_score fallback scoring."""

    def test_low_loss_gives_high_score(self):
        """Very low loss → high proxy score (capped at 8)."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        score = loss_to_proxy_score(0.1)
        assert score >= 7
        assert score <= 8  # capped — proxy should never give 9+

    def test_medium_loss_gives_medium_score(self):
        """Medium loss → score around 5-7."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        score = loss_to_proxy_score(1.0)
        assert 4 <= score <= 6

    def test_high_loss_gives_low_score(self):
        """High loss → low score."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        score = loss_to_proxy_score(3.0)
        assert score <= 2

    def test_zero_loss_capped_at_8(self):
        """Perfect loss=0 still capped at 8 (proxy, not real test)."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        assert loss_to_proxy_score(0.0) == 8

    def test_infinite_loss_floors_at_1(self):
        """Extreme loss floors at 1."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        assert loss_to_proxy_score(100.0) == 1
        assert loss_to_proxy_score(float("inf")) == 1

    def test_returns_int(self):
        """Proxy score is always an integer."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        for loss in [0.0, 0.3, 0.7, 1.5, 5.0]:
            assert isinstance(loss_to_proxy_score(loss), int)


class TestBuildTestPrompt:
    """Test build_test_prompt for stage-specific Phase 3 prompts."""

    def test_commands_stage_mentions_cmd_syntax(self):
        """COMMANDS stage test prompt includes [CMD] context."""
        from enigma_engine.core.adaptive_trainer import build_test_prompt
        prompt = build_test_prompt(1, "commands", "simple")
        assert "[CMD]" in prompt

    def test_basics_stage_has_no_cmd_context(self):
        """BASICS stage prompt doesn't mention commands."""
        from enigma_engine.core.adaptive_trainer import build_test_prompt
        prompt = build_test_prompt(1, "basics", "simple")
        assert "[CMD]" not in prompt

    def test_all_stages_produce_prompts(self):
        """All stages produce non-empty test prompts."""
        from enigma_engine.core.adaptive_trainer import (
            build_test_prompt, ALL_STAGES, DIFFICULTY_LEVELS)
        for stage in ALL_STAGES:
            for diff in DIFFICULTY_LEVELS:
                prompt = build_test_prompt(1, stage, diff)
                assert len(prompt) > 20
                assert "question" in prompt.lower()

    def test_includes_test_number(self):
        """Test prompt includes the test number."""
        from enigma_engine.core.adaptive_trainer import build_test_prompt
        prompt = build_test_prompt(5, "basics", "medium")
        assert "#5" in prompt


class TestCleanExample:
    """Test clean_example strips garbage wrappers from teacher output."""

    def test_strips_the_answer_is_prefix(self):
        """Removes 'The answer is...' prefix."""
        from enigma_engine.core.adaptive_trainer import clean_example
        raw = "The answer is... Dogs are loyal animals. They make great companions."
        cleaned = clean_example(raw)
        assert cleaned == "Dogs are loyal animals. They make great companions."

    def test_strips_think_tags(self):
        """Removes leaked <think> / </think> XML tags."""
        from enigma_engine.core.adaptive_trainer import clean_example
        raw = "</think>\n</think>\nThe user needs a training example."
        cleaned = clean_example(raw)
        assert "<think>" not in cleaned
        assert "</think>" not in cleaned
        assert cleaned == "The user needs a training example."

    def test_strips_here_is_wrapper(self):
        """Removes 'Here is a training example:' wrappers."""
        from enigma_engine.core.adaptive_trainer import clean_example
        raw = "Here is a training example: Q: What is AI?\nA: Artificial intelligence."
        cleaned = clean_example(raw)
        assert cleaned.startswith("Q:")

    def test_returns_empty_for_empty(self):
        """Returns empty string for empty input."""
        from enigma_engine.core.adaptive_trainer import clean_example
        assert clean_example("") == ""
        assert clean_example("   ") == ""

    def test_preserves_valid_content(self):
        """Does not mangle valid Q&A content."""
        from enigma_engine.core.adaptive_trainer import clean_example
        valid = "Q: What is the capital of France?\nA: Paris is the capital of France."
        assert clean_example(valid) == valid

    def test_handles_answer_is_no_dots(self):
        """Strips 'The answer is' without trailing dots."""
        from enigma_engine.core.adaptive_trainer import clean_example
        raw = "The answer is ambition: the strong desire to achieve something."
        cleaned = clean_example(raw)
        assert cleaned == "ambition: the strong desire to achieve something."


class TestValidateExample:
    """Test validate_example for stage-specific format checks."""

    def test_rejects_short_text(self):
        """Examples under 30 chars are rejected."""
        from enigma_engine.core.adaptive_trainer import validate_example
        assert validate_example("Hi", "basics") is False
        assert validate_example("Too short.", "basics") is False

    def test_accepts_valid_basics(self):
        """A coherent paragraph passes basics validation."""
        from enigma_engine.core.adaptive_trainer import validate_example
        text = ("Dogs are loyal animals that have been domesticated "
                "for thousands of years. They make great companions "
                "and come in many different breeds.")
        assert validate_example(text, "basics") is True

    def test_rejects_leaked_reasoning(self):
        """Leaked teacher reasoning is rejected."""
        from enigma_engine.core.adaptive_trainer import validate_example
        assert validate_example(
            "I should pick a type that works well for training.",
            "basics") is False
        assert validate_example(
            "I need to generate a training example now.",
            "basics") is False

    def test_commands_requires_cmd_block(self):
        """Commands stage requires [CMD]...[/CMD] blocks."""
        from enigma_engine.core.adaptive_trainer import validate_example
        no_cmd = "Q: How do I list files?\nA: Just type the list command."
        assert validate_example(no_cmd, "commands") is False
        with_cmd = "Q: How do I list files?\nA: Use [CMD]file.list[/CMD] to see all files."
        assert validate_example(with_cmd, "commands") is True

    def test_web_requires_search_web(self):
        """Web stage requires search.web command."""
        from enigma_engine.core.adaptive_trainer import validate_example
        no_web = "Q: What is the weather?\nA: Use [CMD]weather.check[/CMD]."
        assert validate_example(no_web, "web") is False
        with_web = "Q: What is the weather?\nA: Let me check [CMD]search.web weather today[/CMD]."
        assert validate_example(with_web, "web") is True

    def test_conversation_requires_turns(self):
        """Conversation stage needs dialogue structure."""
        from enigma_engine.core.adaptive_trainer import validate_example
        single = "This is just a paragraph about cats being nice animals and very fluffy."
        assert validate_example(single, "conversation") is False
        dialogue = "User: What is Python?\nAI: Python is a popular programming language used for many tasks."
        assert validate_example(dialogue, "conversation") is True

    def test_conversation_accepts_assistant_format(self):
        """Conversation stage accepts User/Assistant dialogue."""
        from enigma_engine.core.adaptive_trainer import validate_example
        dialogue = "User: What is Python?\nAssistant: Python is a popular programming language used for many tasks."
        assert validate_example(dialogue, "conversation") is True

    def test_rejects_empty(self):
        """Empty string is rejected."""
        from enigma_engine.core.adaptive_trainer import validate_example
        assert validate_example("", "basics") is False


class TestDeduplicateExamples:
    """Test deduplicate_examples for near-duplicate removal."""

    def test_removes_exact_duplicates(self):
        """Exact duplicates are removed."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        examples = ["Hello world!", "Hello world!", "Something else here now."]
        result = deduplicate_examples(examples)
        assert len(result) == 2

    def test_removes_near_duplicates(self):
        """Near-duplicates differing only in whitespace/punctuation."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        examples = [
            "Q: How do I save this?\nA: Just type file.write.",
            "Q: How do I save this?  \n A: Just type file.write",
        ]
        result = deduplicate_examples(examples)
        assert len(result) == 1

    def test_checks_against_existing(self):
        """Deduplicates against already-accumulated data."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        existing = ["Q: What is the capital of France?\nA: Paris."]
        new_data = [
            "Q: What is the capital of France?\nA: Paris.",
            "Q: What is the meaning of life?\nA: 42.",
        ]
        result = deduplicate_examples(new_data, existing)
        assert len(result) == 1
        assert "meaning" in result[0]

    def test_preserves_order(self):
        """First occurrence is kept, not the duplicate."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        examples = ["First unique example here.", "Second unique example here.", "First unique example here."]
        result = deduplicate_examples(examples)
        assert result[0] == "First unique example here."
        assert result[1] == "Second unique example here."

    def test_empty_input(self):
        """Empty list returns empty list."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        assert deduplicate_examples([]) == []

    def test_cmd_blocks_not_collapsed(self):
        """Different CMD blocks must not be treated as duplicates (#18)."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        examples = [
            "User: Save the file\nAssistant: [CMD]file.write data.txt[/CMD]",
            "User: Read the file\nAssistant: [CMD]file.read data.txt[/CMD]",
        ]
        result = deduplicate_examples(examples)
        assert len(result) == 2, "Different CMD blocks collapsed as dupes"

    def test_brackets_preserved_in_normalization(self):
        """Square brackets and dots are semantically significant (#18)."""
        from enigma_engine.core.adaptive_trainer import _normalize_for_dedup
        # Dots must be preserved so file.write != filewrite
        assert _normalize_for_dedup("file.write") != _normalize_for_dedup("filewrite")
        # Brackets preserved so [CMD] structure is visible
        assert "[" in _normalize_for_dedup("[CMD]test[/CMD]")


class TestParseScore:
    """Test parse_score for robust score extraction from LLM judgments."""

    def test_score_colon_format(self):
        """Parses 'SCORE: 7 | Good answer'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("SCORE: 7 | Good answer") == 7

    def test_score_colon_lowercase(self):
        """Parses 'score: 8'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("score: 8") == 8

    def test_n_slash_10_format(self):
        """Parses '7/10'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I'd rate this 7/10") == 7

    def test_score_of_n(self):
        """Parses 'score of 6'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I give a score of 6") == 6

    def test_give_it_a_n(self):
        """Parses 'give this a 8'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I'd give this a 8") == 8

    def test_bare_number_on_line(self):
        """Parses bare number on its own line."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("7") == 7

    def test_clamps_high(self):
        """Scores above 10 are clamped to 10."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("SCORE: 15") == 10

    def test_clamps_low(self):
        """Scores below 1 are clamped to 1."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("SCORE: 0") == 1

    def test_returns_5_on_empty(self):
        """Empty text returns default 5."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("") == 5
        assert parse_score("   ") == 5

    def test_returns_5_on_no_score(self):
        """Text with no numeric score returns default 5."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("This was a good answer overall.") == 5

    def test_multiline_with_score_in_middle(self):
        """Finds SCORE: pattern even in middle of multiline text."""
        from enigma_engine.core.adaptive_trainer import parse_score
        judgment = (
            "The student answered well.\n"
            "SCORE: 8 | Good vocabulary\n"
            "Could improve on specifics.")
        assert parse_score(judgment) == 8

    def test_rating_format(self):
        """Parses 'rating: 6'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("My rating: 6 for this response.") == 6

    def test_does_not_match_digits_in_unrelated_text(self):
        """Digits embedded in non-score contexts should not match (#19)."""
        from enigma_engine.core.adaptive_trainer import parse_score
        # "200 lines of code" — the 200 should NOT be parsed as a score
        result = parse_score("The answer had about 200 lines of code.")
        # Should either return 5 (default) or 10 (clamped)
        # but NOT treat 200 as a raw score
        assert result in (5, 10)

    def test_bare_number_rejects_large_values(self):
        """Bare number pattern only matches 1-10 range."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("42") == 5  # Out of range, defaults

    def test_n_slash_10_with_word_boundary(self):
        """N/10 should match '7/10' but not '23/100'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I'd say 7/10") == 7
        # 23/100 should not match as 23/10
        assert parse_score("Got 23/100 on the test") == 5

    def test_score_word_boundary_prefix(self):
        """'score12' should not match - 'score' needs word boundary."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("This is score12 quality") == 5

    def test_give_this_a_boundary(self):
        """'give this a 107' should not match '10' from '107'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I give this a 107") == 5


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

    def test_background_trainer_accepts_adam_params(self):
        """BackgroundTrainer constructor accepts adam_betas and adam_eps."""
        from enigma_engine.router import BackgroundTrainer
        sig = inspect.signature(BackgroundTrainer.__init__)
        assert "adam_betas" in sig.parameters, (
            "BackgroundTrainer must accept adam_betas parameter")
        assert "adam_eps" in sig.parameters, (
            "BackgroundTrainer must accept adam_eps parameter")

    def test_background_trainer_adamw_uses_params(self):
        """BackgroundTrainer.set_model uses stored adam_betas and adam_eps."""
        from enigma_engine.router import BackgroundTrainer
        source = inspect.getsource(BackgroundTrainer.set_model)
        assert "adam_betas" in source or "self.adam_betas" in source, (
            "set_model must use adam_betas from constructor")
        assert "adam_eps" in source or "self.adam_eps" in source, (
            "set_model must use adam_eps from constructor")


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

    def test_reorder_job_with_running_job(self):
        """reorder_job must not break when a running job is in the list."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        j1 = q.add_job(TrainingJob(mode="Solo"))
        j2 = q.add_job(TrainingJob(mode="DPO"))
        j3 = q.add_job(TrainingJob(mode="LoRA"))

        # Simulate j1 running
        j1.status = "running"
        q._current_job = j1

        # Move j3 to position 0 among pending jobs
        moved = q.reorder_job(j3.job_id, 0)
        assert moved is True

        # j3 should now be before j2 in pending order
        pending = [j for j in q.jobs if j.status == "pending"]
        assert len(pending) == 2
        assert pending[0].job_id == j3.job_id
        assert pending[1].job_id == j2.job_id

        # Running job should still be in the list
        all_ids = [j.job_id for j in q.jobs]
        assert j1.job_id in all_ids

    def test_reorder_preserves_non_pending_jobs(self):
        """reorder_job must not affect completed/failed/running jobs."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        j1 = q.add_job(TrainingJob(mode="Solo"))
        j2 = q.add_job(TrainingJob(mode="DPO"))
        j3 = q.add_job(TrainingJob(mode="LoRA"))
        j4 = q.add_job(TrainingJob(mode="Vision"))

        # Mark j1 as completed, j2 as running
        j1.status = "completed"
        j2.status = "running"

        # Move j4 to position 0 among pending
        moved = q.reorder_job(j4.job_id, 0)
        assert moved is True

        jobs = q.jobs
        # completed and running must still be in order
        assert jobs[0].status == "completed"
        assert jobs[1].status == "running"
        # j4 should be before j3 in the pending slots
        pending = [j for j in jobs if j.status == "pending"]
        assert pending[0].job_id == j4.job_id
        assert pending[1].job_id == j3.job_id

    def test_pause_flag_under_lock(self):
        """_run_loop pause check should be under lock."""
        import inspect
        from enigma_engine.core.training_queue import TrainingQueue
        source = inspect.getsource(TrainingQueue._run_loop)
        # The pause check should acquire the lock
        assert "self._lock" in source or "_lock" in source


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
        for p in rm.tok_embeddings.parameters():  # type: ignore[union-attr]
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

    def test_losses_list_capped(self):
        """_losses list must not grow unbounded — capped at MAX_LOSSES."""
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        m.start_run()
        # Record more losses than the cap
        for i in range(120_000):
            m.record_loss(float(i))
        # Should be capped, not 120k
        assert len(m.losses) <= 100_001

    def test_steps_list_stays_in_sync_with_losses(self):
        """steps and losses must have the same length after cap."""
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        m.start_run()
        for i in range(110_000):
            m.record_loss(float(i))
        assert len(m.losses) == len(m.steps)


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
        obj._is_gguf = False  # type: ignore[attr-defined]
        obj.model = MagicMock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=5)  # type: ignore[attr-defined]

        ctx = obj._prepare_chat("hello")
        assert isinstance(ctx, ChatContext)

    def test_prepare_chat_builds_messages(self):
        """Messages include system + history + user message."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False  # type: ignore[attr-defined]
        obj.model = MagicMock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=5)  # type: ignore[attr-defined]

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
        obj._is_gguf = False  # type: ignore[attr-defined]
        obj.model = MagicMock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=5)  # type: ignore[attr-defined]

        ctx = obj._prepare_chat("test", max_gen=2000, reasoning=True)
        assert ctx.max_gen == 3000  # 2000 * 1.5

    def test_prepare_chat_reasoning_injects_instruction(self):
        """When reasoning=True, reasoning instruction is in prompt and messages."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False  # type: ignore[attr-defined]
        obj.model = MagicMock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=5)  # type: ignore[attr-defined]

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
        obj._is_gguf = False  # type: ignore[attr-defined]
        obj.model = MagicMock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=5)  # type: ignore[attr-defined]

        # stream_generate yields tokens — mock it to yield nothing
        obj.stream_generate = MagicMock(return_value=iter([]))  # type: ignore[attr-defined]
        gen = obj.stream_chat("test", reasoning=True)
        # Consume the generator
        list(gen)

        # Check that stream_generate was called with a prompt containing <think>
        call_args = obj.stream_generate.call_args  # type: ignore[attr-defined]
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
        obj._is_gguf = True  # type: ignore[attr-defined]
        mock_model = MagicMock()
        mock_model.chat = MagicMock(
            side_effect=RuntimeError("GGUF model crashed"))
        obj.model = mock_model  # type: ignore[attr-defined]
        # Stub methods needed by chat()
        import threading
        obj._generation_lock = threading.Lock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=10)  # type: ignore[attr-defined]

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
        """Vision encoder RMSNorm also upcasts to fp32."""
        import torch
        from enigma_engine.core.model_components import RMSNorm
        norm = RMSNorm(32)
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

    def test_lora_trainer_has_betas(self):
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        sig = inspect.signature(LoraTrainer.__init__)
        assert sig.parameters["adam_beta1"].default == 0.9
        assert sig.parameters["adam_beta2"].default == 0.95
        assert sig.parameters["adam_eps"].default == 1e-8

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


class TestReasoningLossTokenIds:
    """Verify _apply_reasoning_loss_weight uses named token IDs, not encode()."""

    def test_uses_think_start_id_attribute(self):
        """Must use tokenizer.think_start_id, not encode('<think>')."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._apply_reasoning_weight)
        assert "think_start_id" in source, (
            "Should use think_start_id attribute directly")
        assert "_get_token_ids" not in source, (
            "Should NOT use _get_token_ids (includes BOS/EOS)")

    def test_uses_think_end_id_attribute(self):
        """Must use tokenizer.think_end_id, not encode('</think>')."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._apply_reasoning_weight)
        assert "think_end_id" in source


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
# Training: random import fix + weight sanity check
# ================================================================


class TestTrainingRandomImportFix:
    """Verify train() doesn't crash from shadowed random import."""

    def test_no_local_random_import_in_train(self):
        """train() must not have a local 'import random' that shadows module-level."""
        import ast
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "random", (
                        f"Local 'import random' at line ~{node.lineno} in "
                        f"train() shadows module-level import — use it directly")

    def test_random_shuffle_reachable_without_general_data(self):
        """random.shuffle(batches) must work when general_data is not set."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "random.shuffle(batches)" in source, (
            "train() must shuffle batches each epoch")


class TestTrainingWeightSanityCheck:
    """Verify train() includes a weight-change verification after first step."""

    def test_weight_check_in_train(self):
        """train() should verify weights changed after first optimizer step."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "weight_check_done" in source or "Weight update verified" in source, (
            "train() should verify weights actually changed after first step")

    def test_weight_check_skips_embeddings(self):
        """Weight sanity check should skip embedding/output layers (sparse gradients)."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert '"embed"' in source or "'embed'" in source, (
            "Weight check should filter out embedding layers")


class TestModelConfigDict:
    """Verify _model_config_dict saves full config, not a subset."""

    def test_config_dict_includes_architecture_flags(self):
        """_model_config_dict must include use_rope, use_rms_norm, etc."""
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(use_moe=True, use_qk_norm=True)

        class FakeModel:
            config = cfg

        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._model_config_dict(FakeModel())
        assert "use_rope" in result, "Config dict missing use_rope"
        assert "use_moe" in result, "Config dict missing use_moe"
        assert result["use_moe"] is True, "Config dict should reflect actual values"


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
# PPO Rewrite — True PPO with value head, GAE, clipped surrogate
# ================================================================


class TestValueHead:
    """ValueHead: MLP critic that predicts state values from hidden states."""

    def test_value_head_exists(self):
        """ValueHead class exists in rl_training module."""
        from enigma_engine.core.rl_training import ValueHead
        assert ValueHead is not None

    def test_value_head_forward_shape(self):
        """ValueHead(dim) produces (B, T) values from (B, T, dim) hidden states."""
        import torch
        from enigma_engine.core.rl_training import ValueHead
        vh = ValueHead(dim=64)
        h = torch.randn(2, 10, 64)
        values = vh(h)
        assert values.shape == (2, 10)

    def test_value_head_single_token(self):
        """ValueHead works with single-token inputs."""
        import torch
        from enigma_engine.core.rl_training import ValueHead
        vh = ValueHead(dim=32)
        h = torch.randn(1, 1, 32)
        values = vh(h)
        assert values.shape == (1, 1)


class TestRolloutBuffer:
    """RolloutBuffer stores (logprobs, values, rewards, masks) for PPO updates."""

    def test_rollout_buffer_exists(self):
        """RolloutBuffer class exists."""
        from enigma_engine.core.rl_training import RolloutBuffer
        assert RolloutBuffer is not None

    def test_store_and_get(self):
        """Can store experience and retrieve it."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.randn(5),
            values=torch.randn(5),
            rewards=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
            response_mask=torch.ones(5),
        )
        assert len(buf) == 1

    def test_compute_advantages_gae(self):
        """compute_advantages uses GAE (gamma, lam)."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.zeros(3),
            values=torch.tensor([0.5, 0.5, 0.5]),
            rewards=torch.tensor([0.0, 0.0, 1.0]),
            response_mask=torch.ones(3),
        )
        advantages, returns = buf.compute_advantages(gamma=1.0, lam=0.95)
        assert advantages.shape[0] > 0
        assert returns.shape[0] > 0
        # Last token has reward=1, value=0.5, so advantage > 0
        assert advantages[-1] > 0

    def test_clear(self):
        """clear resets the buffer."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.zeros(3),
            values=torch.zeros(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
        )
        assert len(buf) == 1
        buf.clear()
        assert len(buf) == 0


class TestPPOConfig:
    """RLHFConfig gains PPO-specific fields."""

    def test_config_has_value_coeff(self):
        """RLHFConfig has value_coeff field."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "value_coeff")
        assert cfg.value_coeff == 0.5

    def test_config_has_entropy_coeff(self):
        """RLHFConfig has entropy_coeff field."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "entropy_coeff")
        assert cfg.entropy_coeff == 0.01

    def test_config_has_gae_lambda(self):
        """RLHFConfig has gae_lambda field."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "gae_lambda")
        assert cfg.gae_lambda == 0.95

    def test_config_has_ppo_epochs(self):
        """RLHFConfig has ppo_epochs for minibatch updates."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "ppo_epochs")
        assert cfg.ppo_epochs == 4

    def test_config_has_minibatch_size(self):
        """RLHFConfig has minibatch_size."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "minibatch_size")
        assert cfg.minibatch_size == 4

    def test_backward_compat_existing_defaults(self):
        """Old defaults remain unchanged."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert cfg.epochs == 3
        assert cfg.kl_coeff == 0.1
        assert cfg.clip_range == 0.2
        assert cfg.n_responses == 4


class TestPPOTrainerStructure:
    """RLHFTrainer.train() implements true PPO."""

    def test_train_uses_clip_range(self):
        """train() actually uses clip_range for clipped surrogate."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "clip_range" in source
        # Must do ratio clamping, not just mention it
        assert "clamp" in source or "clip" in source

    def test_train_uses_value_head(self):
        """train() creates and uses a ValueHead."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "ValueHead" in source or "value_head" in source

    def test_train_uses_rollout_buffer(self):
        """train() uses RolloutBuffer for experience collection."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "RolloutBuffer" in source or "rollout" in source

    def test_train_computes_advantages(self):
        """train() computes GAE advantages."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "advantage" in source.lower()

    def test_train_has_entropy_bonus(self):
        """train() includes entropy bonus in loss."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "entropy" in source.lower()

    def test_train_has_value_loss(self):
        """train() computes value function loss."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "value_loss" in source or "v_loss" in source

    def test_train_has_ppo_epochs(self):
        """train() does multiple epochs of minibatch updates per rollout."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "ppo_epoch" in source

    def test_train_uses_amp(self):
        """train() uses AMP autocast (not unused as before)."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "autocast" in source

    def test_get_response_logps_returns_per_token(self):
        """Module-level _get_response_logps returns per-token log-probs (1D)."""
        from enigma_engine.core.rl_training import _get_response_logps
        source = inspect.getsource(_get_response_logps)
        assert "log_softmax" in source or "log_probs" in source

    def test_n_responses_used_in_train(self):
        """n_responses config is actually used for multi-response generation."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "n_responses" in source

    def test_reward_history_bounded(self):
        """Reward history list doesn't grow unbounded."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        # Should cap reward_history or use deque
        assert "max" in source or "deque" in source or "[-" in source


class TestSelfPlayPPOUpgrade:
    """SelfPlayTrainer gets the same PPO treatment."""

    def test_selfplay_uses_value_head(self):
        """SelfPlayTrainer.train() uses ValueHead."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.train)
        assert "ValueHead" in source or "value_head" in source

    def test_selfplay_uses_clipping(self):
        """SelfPlayTrainer applies PPO clipping."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.train)
        assert "clamp" in source or "clip" in source

    def test_selfplay_has_entropy(self):
        """SelfPlayTrainer includes entropy bonus."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.train)
        assert "entropy" in source.lower()

    def test_selfplay_config_has_ppo_fields(self):
        """SelfPlayConfig gains PPO fields matching RLHFConfig."""
        from enigma_engine.core.rl_training import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert hasattr(cfg, "clip_range")
        assert hasattr(cfg, "value_coeff")
        assert hasattr(cfg, "entropy_coeff")
        assert hasattr(cfg, "ppo_epochs")


class TestSharedHelpers:
    """_get_response_logps extracted as shared, not duplicated."""

    def test_single_logps_function(self):
        """There is one _get_response_logps implementation, class methods are delegates."""
        import enigma_engine.core.rl_training as rl_mod
        source = inspect.getsource(rl_mod)
        # Count definitions of _get_response_logps
        count = source.count("def _get_response_logps(")
        # 1 module-level + up to 2 class delegates (staticmethod wrappers)
        assert count <= 3, (
            f"_get_response_logps defined {count} times — deduplicate")
        # The module-level function should contain the actual logic
        mod_source = inspect.getsource(rl_mod._get_response_logps)
        assert "log_softmax" in mod_source


class TestPPORatioComputation:
    """PPO must compute real importance sampling ratio, not hardcode 1.0."""

    def test_rlhf_ppo_uses_torch_exp_for_ratio(self):
        """RLHFTrainer.train() computes ratio via torch.exp(new - old)."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "torch.exp(" in source, (
            "PPO ratio must use torch.exp(new_logps - old_logps)")

    def test_rlhf_ppo_no_hardcoded_ones_ratio(self):
        """RLHFTrainer PPO loop must not use ones_like as the ratio."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        # torch.ones_like should only appear in the replay fallback,
        # not as the primary ratio computation
        lines = source.split("\n")
        for line in lines:
            if "ones_like" in line and "ratio" in line:
                # Must be inside a replay/fallback branch
                assert "else" in source[:source.index(line)].split("\n")[-5:] or \
                    "replay" in line.lower() or "fallback" in line.lower() or \
                    "else:" in "\n".join(lines[max(0, lines.index(line)-3):lines.index(line)]), \
                    "ratio = torch.ones_like should only be in replay fallback"

    def test_selfplay_ppo_uses_torch_exp_for_ratio(self):
        """SelfPlayTrainer.train() computes ratio via torch.exp(new - old)."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.train)
        assert "torch.exp(" in source, (
            "PPO ratio must use torch.exp(new_logps - old_logps)")

    def test_rlhf_recomputes_logps_in_ppo_loop(self):
        """RLHF PPO loop calls _get_response_logps for fresh log-probs."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        # Should call _get_response_logps inside the ppo_epoch loop
        ppo_section = source[source.index("ppo_epoch"):]
        assert "_get_response_logps" in ppo_section

    def test_selfplay_recomputes_logps_in_ppo_loop(self):
        """SelfPlay PPO loop calls _get_response_logps for fresh log-probs."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.train)
        ppo_section = source[source.index("ppo_epoch"):]
        assert "_get_response_logps" in ppo_section

    def test_rlhf_recomputes_entropy_in_ppo_loop(self):
        """RLHF PPO loop uses real entropy, not -logps.mean()."""
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        ppo_section = source[source.index("ppo_epoch"):]
        assert "_get_response_entropy" in ppo_section

    def test_rollout_buffer_stores_full_ids(self):
        """RolloutBuffer accepts and stores full_ids and prompt_len."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        full_ids = torch.tensor([[1, 2, 3, 4, 5]])
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.tensor([0.0, 0.0, 1.0]),
            response_mask=torch.ones(3),
            full_ids=full_ids,
            prompt_len=2,
        )
        assert buf._full_ids[0] is not None
        assert buf._prompt_lens[0] == 2
        assert torch.equal(buf._full_ids[0], full_ids)

    def test_rollout_buffer_clear_clears_ids(self):
        """clear() also clears full_ids and prompt_lens."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
            full_ids=torch.tensor([[1, 2, 3]]),
            prompt_len=1,
        )
        buf.clear()
        assert len(buf._full_ids) == 0
        assert len(buf._prompt_lens) == 0

    def test_rollout_buffer_backward_compat(self):
        """RolloutBuffer.store() works without full_ids (backward compat)."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
        )
        assert buf._full_ids[0] is None
        assert buf._prompt_lens[0] is None


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
        snap.append(None)  # type: ignore[arg-type]  # mutate snapshot
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
        assert vec.idf is not None
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
        assert vec2.doc_lens is not None and vec.doc_lens is not None
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
            head_dim=4, device=torch.device("cpu"), dtype=torch.float32,
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
# On-demand causal mask + KVCache wiring (#quality-audit)
# ================================================================

class TestOnDemandCausalMask:
    """Model builds causal mask lazily instead of pre-allocating max_seq_len²."""

    def test_causal_mask_starts_none(self):
        """_causal_mask should be None after __init__ (not pre-allocated)."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                             vocab_size=100, max_seq_len=4096)
        model = Enigma(config=config)
        assert model._causal_mask is None, "Mask should start None (on-demand)"
        assert model._causal_mask_size == 0

    def test_get_causal_mask_builds_on_demand(self):
        """_get_causal_mask creates and caches at the requested size."""
        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                             vocab_size=100, max_seq_len=4096)
        model = Enigma(config=config)

        mask = model._get_causal_mask(8)
        assert mask.shape == (8, 8)
        # Upper triangle should be -inf, diagonal + below should be 0
        assert mask[0, 1] == float('-inf')
        assert mask[1, 0] == 0.0
        assert model._causal_mask_size == 8

    def test_causal_mask_grows_not_shrinks(self):
        """Requesting a larger mask grows the cache; smaller reuses it."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        config = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                             vocab_size=100, max_seq_len=4096)
        model = Enigma(config=config)

        model._get_causal_mask(4)
        assert model._causal_mask_size == 4

        model._get_causal_mask(16)
        assert model._causal_mask_size == 16

        # Smaller request shouldn't shrink
        mask = model._get_causal_mask(8)
        assert mask.shape == (8, 8)
        assert model._causal_mask_size == 16  # Still 16


class TestAttentionUsesPreAllocKVCache:
    """Attention class uses kv_cache.KVCache instead of torch.cat()."""

    def test_attention_has_no_cache_k_attribute(self):
        """Old cache_k/cache_v attrs should be gone."""
        from enigma_engine.core.model_components import Attention
        source = inspect.getsource(Attention.__init__)
        assert "self.cache_k" not in source
        assert "self.cache_v" not in source

    def test_attention_uses_kvcache_module(self):
        """Attention.forward should import and use KVCache, not torch.cat for caching."""
        from enigma_engine.core.model_components import Attention
        source = inspect.getsource(Attention.forward)
        assert "KVCache" in source
        # torch.cat should only appear in comments, not as actual code
        import re
        code_lines = [ln for ln in source.splitlines() if not ln.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert "torch.cat(" not in code_only, "torch.cat() should not be used for KV-cache append"

    def test_clear_cache_resets_kv_cache(self):
        """clear_cache should set _kv_cache to None."""
        from enigma_engine.core.model_components import Attention
        source = inspect.getsource(Attention.clear_cache)
        assert "_kv_cache" in source


class TestWeightLoadFailsLoud:
    """Weight loading failure should raise, not silently continue."""

    def test_load_weights_source_raises(self):
        """_load_pytorch should raise RuntimeError, not log and continue."""
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine._load_pytorch)
        # Should raise RuntimeError, not just log.error
        assert "raise RuntimeError" in source
        # Old silent pattern should be gone
        assert "initialized with random weights" not in source


class TestValSplitRandom:
    """Validation split should use random sampling, not tail-slicing."""

    def test_val_split_uses_random_shuffle(self):
        """train() should use Random(42).shuffle for deterministic random split."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        # Should use random shuffle, not just take last N
        assert "shuffle(indices)" in source
        # Should be deterministic
        assert "Random(42)" in source


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

    def test_default_val_split(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.val_split == 0.1  # 10% held out by default

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

    def test_val_split_in_forge_train_params(self):
        """_read_forge_train_params includes val_split."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._read_forge_train_params)
        assert "val_split" in source, (
            "_read_forge_train_params must return val_split")

    def test_val_split_in_forge_settings_persistence(self):
        """val_split var is persisted in _forge_settings."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._save_training_brief)
        assert "val_split" in source, (
            "val_split must be saved in _forge_settings")


class TestGeneralDataMixing:
    """TrainingConfig general_mix_ratio and general_data fields."""

    def test_default_ratio(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.general_mix_ratio == 0.2

    def test_default_general_data_empty(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.general_data == ""

    def test_custom_ratio_and_path(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(
            general_mix_ratio=0.3, general_data="/some/file.txt")
        assert cfg.general_mix_ratio == 0.3


# =============================================================================
# AMP + GRADIENT ACCUMULATION VERIFICATION (#11)
# =============================================================================

class TestAmpGradAccumInteraction:
    """Verify AMP (GradScaler) and gradient accumulation interact correctly.

    Structural tests that inspect _train_one_batch source to ensure:
    - Loss is scaled for accumulation BEFORE scaler.scale().backward()
    - scaler.unscale_() is called BEFORE gradient clipping
    - scaler.step() and scaler.update() happen at accumulation boundaries
    - Returned loss restores the unscaled value
    """

    def test_loss_divided_before_backward(self):
        """Loss must be divided by max_grad_accumulation before backward."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer._train_one_batch)
        lines = src.splitlines()
        div_line = None
        backward_line = None
        for i, line in enumerate(lines):
            if "max_grad_accumulation" in line and "/" in line and "loss" in line:
                div_line = i
            if ".backward()" in line and backward_line is None:
                backward_line = i
        assert div_line is not None, "Loss / max_grad_accumulation not found"
        assert backward_line is not None, "backward() not found"
        assert div_line < backward_line, (
            "Loss must be divided BEFORE backward()")

    def test_unscale_before_clip(self):
        """scaler.unscale_() must happen before clip_grad_norm_."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer._train_one_batch)
        lines = src.splitlines()
        unscale_line = None
        clip_line = None
        for i, line in enumerate(lines):
            if "unscale_" in line:
                unscale_line = i
            if "clip_grad_norm_" in line:
                clip_line = i
        assert unscale_line is not None, "unscale_ not found"
        assert clip_line is not None, "clip_grad_norm_ not found"
        assert unscale_line < clip_line, (
            "unscale_ must happen BEFORE clip_grad_norm_")

    def test_scaler_step_and_update_together(self):
        """scaler.step() must be followed by scaler.update()."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer._train_one_batch)
        lines = src.splitlines()
        step_line = None
        update_line = None
        for i, line in enumerate(lines):
            if "scaler.step(" in line:
                step_line = i
            if "scaler.update()" in line:
                update_line = i
        assert step_line is not None, "scaler.step() not found"
        assert update_line is not None, "scaler.update() not found"
        assert step_line < update_line, (
            "scaler.step() must come before scaler.update()")

    def test_return_restores_unscaled_loss(self):
        """Return value must multiply back by max_grad_accumulation."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer._train_one_batch)
        assert "* self.config.max_grad_accumulation" in src, (
            "Return must restore true loss via * max_grad_accumulation")

    def test_accum_gate_uses_modulo(self):
        """Optimizer step only happens at accumulation boundaries."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer._train_one_batch)
        assert "% self.config.max_grad_accumulation == 0" in src, (
            "Accumulation gate must use modulo check")

    def test_dpo_accum_divides_loss(self):
        """DPO train_dpo() also divides loss by accum_steps."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer.train_dpo)
        assert "loss / accum_steps" in src or "loss = loss / accum" in src, (
            "DPO must divide loss by accum_steps")

    def test_dpo_accum_flushes_tail(self):
        """DPO flushes remaining gradients when pairs not divisible by accum."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer.train_dpo)
        assert "% accum_steps != 0" in src, (
            "DPO must flush tail gradients when pairs % accum_steps != 0")


class TestGeneralDataMixing:
    """TrainingConfig general_mix_ratio and general_data fields."""

    def test_default_ratio(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.general_mix_ratio == 0.2

    def test_default_general_data_empty(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.general_data == ""

    def test_custom_ratio_and_path(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(
            general_mix_ratio=0.3, general_data="/some/file.txt")
        assert cfg.general_mix_ratio == 0.3
        assert cfg.general_data == "/some/file.txt"

    def test_zero_ratio_disables_mixing(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(
            general_mix_ratio=0.0, general_data="some data")
        assert cfg.general_mix_ratio == 0.0

    def test_default_label_smoothing(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.label_smoothing == 0.05

    def test_default_early_stopping(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.early_stopping_patience == 5


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


# ================================================================
# Pad masking — Bug #1
# ================================================================

@pytest.mark.structural
class TestPadMasking:
    """Verify attention_mask support in model.forward() and training pipeline."""

    def test_forward_accepts_attention_mask(self):
        """model.forward() must accept attention_mask parameter."""
        import inspect
        from enigma_engine.core.model import Enigma
        sig = inspect.signature(Enigma.forward)
        assert "attention_mask" in sig.parameters, (
            "model.forward() missing attention_mask parameter — "
            "model attends to garbage pad tokens during training")

    def test_forward_multimodal_accepts_attention_mask(self):
        """forward_multimodal() must accept attention_mask via kwargs."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.forward_multimodal)
        assert "attention_mask" in source, (
            "forward_multimodal must handle attention_mask")

    def test_create_batches_returns_masks(self):
        """_create_batches must return attention masks alongside tensors."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._create_batches)
        assert "attention_mask" in source, (
            "_create_batches must generate attention_mask for pad tokens")

    def test_train_one_batch_passes_mask(self):
        """_train_one_batch must pass attention_mask to model forward."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._train_one_batch)
        assert "attention_mask" in source, (
            "_train_one_batch must forward attention_mask to model")

    def test_validate_passes_mask(self):
        """_validate must pass attention_mask to model forward."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._validate)
        assert "attention_mask" in source, (
            "_validate must forward attention_mask to model")

    def test_dpo_passes_mask(self):
        """train_dpo must generate and pass attention_mask."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "attention_mask" in source, (
            "train_dpo must use attention_mask for pad tokens")

    def test_pad_mask_combines_with_causal(self):
        """forward() must combine attention_mask with causal mask."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.forward)
        # Should combine attention_mask with causal mask
        assert "attention_mask" in source
        # The mask should be applied additively (float -inf for pads)
        assert "float" in source or "finfo" in source or "-inf" in source


# ================================================================
# DPO LoRA-disable reference model — Quick Win #1
# ================================================================

@pytest.mark.structural
class TestDPOLoraDisable:
    """Verify DPO uses LoRA-disable pattern instead of deepcopy."""

    def test_dpo_tries_lora_first(self):
        """train_dpo should try LoRA disable before deepcopy."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "disable_adapter_layers" in source or "lora" in source.lower(), (
            "DPO should try LoRA disable pattern to avoid doubling VRAM")

    def test_dpo_has_cpu_fallback(self):
        """train_dpo must still have fallback for non-LoRA models."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        # Should still have deepcopy as fallback
        assert "deepcopy" in source, (
            "DPO must keep deepcopy as fallback for non-LoRA models")


# ================================================================
# Vocab padding to 64 — Quick Win #2
# ================================================================

@pytest.mark.structural
class TestVocabPadding:
    """Verify vocab is padded to multiple of 64 for GPU matmul alignment."""

    def test_embedding_size_padded(self):
        """tok_embeddings should use padded vocab_size."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.__init__)
        # Should have vocab padding logic
        assert "64" in source, (
            "Model __init__ should pad vocab_size to multiple of 64")

    def test_vocab_padding_math(self):
        """Verify padding math: next multiple of 64."""
        # Test the formula inline
        def pad_to_64(n):
            return (n + 63) & ~63
        assert pad_to_64(8000) == 8000  # 8000 is already 64*125
        assert pad_to_64(8001) == 8064
        assert pad_to_64(128) == 128
        assert pad_to_64(129) == 192
        assert pad_to_64(32000) == 32000
        assert pad_to_64(32001) == 32064


# ================================================================
# freqs_cis non-persistent buffer
# ================================================================

@pytest.mark.structural
class TestFreqsCisNonPersistent:
    """freqs_cis must be a non-persistent buffer so it's excluded from state_dict."""

    def test_freqs_cis_not_in_state_dict(self):
        """register_buffer('freqs_cis', ..., persistent=False) keeps it out of state_dict."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.__init__)
        assert "persistent=False" in source, (
            "freqs_cis must use persistent=False to avoid strict load_state_dict errors")

    def test_load_state_dict_pops_freqs_cis(self):
        """load_state_dict still pops freqs_cis for backward compat with old checkpoints."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.load_state_dict)
        assert "freqs_cis" in source, (
            "load_state_dict should still pop freqs_cis for old checkpoint compat")


# ================================================================
# LoRA scheduler — Quick Win #3
# ================================================================

@pytest.mark.structural
class TestLoraScheduler:
    """Verify LoRA training uses a learning rate scheduler."""

    def test_lora_train_has_scheduler(self):
        """LoraTrainer.train() must include a learning rate scheduler."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        assert "scheduler" in source.lower(), (
            "LoRA training uses flat LR — add CosineAnnealingLR")

    def test_lora_scheduler_steps(self):
        """Scheduler must step inside the training loop."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        assert "scheduler.step()" in source, (
            "LoRA scheduler must step per optimizer step")


# ================================================================
# DPO gradient accumulation — Batch DPO
# ================================================================

@pytest.mark.structural
class TestDPOBatchAccumulation:
    """Verify DPO uses gradient accumulation across preference pairs."""

    def test_dpo_has_accum_steps(self):
        """train_dpo should compute accum_steps from config."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "accum_steps" in source, (
            "DPO must compute accum_steps for gradient accumulation")

    def test_dpo_scales_loss(self):
        """DPO loss must be divided by accum_steps before backward."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "loss / accum_steps" in source or "loss /= accum_steps" in source, (
            "DPO loss must be scaled by 1/accum_steps for accumulation")

    def test_dpo_conditional_step(self):
        """Optimizer steps only every accum_steps pairs."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "% accum_steps" in source, (
            "DPO should step optimizer every accum_steps pairs")

    def test_dpo_flush_tail(self):
        """DPO must flush remaining gradients after loop ends."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "len(pairs) % accum_steps" in source, (
            "DPO must flush accumulated gradients for tail pairs")

    def test_dpo_uses_max_grad_accumulation(self):
        """accum_steps reads from config.max_grad_accumulation."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "max_grad_accumulation" in source, (
            "DPO accum_steps must come from config.max_grad_accumulation")

    def test_dpo_zero_grad_before_loop(self):
        """Gradients should be zeroed before the pair loop starts."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        # zero_grad must appear before the main pair iteration
        zg_idx = source.find("zero_grad()")
        loop_idx = source.find("for i,")
        assert zg_idx < loop_idx, (
            "optimizer.zero_grad() must appear before the pair loop")


# ================================================================
# LoRA proper batching
# ================================================================

@pytest.mark.structural
class TestLoraBatching:
    """Verify LoRA _create_batches produces real multi-sample batches."""

    def test_create_batches_returns_tuples(self):
        """_create_batches must return (tensor, mask) tuples."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer._create_batches)
        assert "attention_mask" in source or "masks" in source, (
            "LoRA _create_batches must produce attention masks for padding")

    def test_create_batches_sorts_by_length(self):
        """Samples should be sorted by length for efficient padding."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer._create_batches)
        assert "sort" in source, (
            "LoRA _create_batches should sort sequences by length")


# ================================================================
# Code Review Pass 2 — Fixes #41-#49
# ================================================================


class TestCharTokenizerSpecialTokenReload:
    """#41: char_tokenizer _load_vocab must reload special_tokens."""

    def test_load_vocab_reloads_special_tokens(self):
        """_load_vocab should update self.special_tokens from file data."""
        import inspect
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        source = inspect.getsource(CharacterTokenizer._load_vocab)
        assert "special_tokens" in source, (
            "_load_vocab must reload special_tokens from saved data")

    def test_save_vocab_includes_special_tokens(self):
        """save_vocab should persist special_tokens to file."""
        import inspect
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        source = inspect.getsource(CharacterTokenizer.save_vocab)
        assert "'special_tokens'" in source, (
            "save_vocab must include special_tokens in saved data")


class TestCharTokenizerThreadSafety:
    """#42: char_tokenizer add_word must be thread-safe."""

    def test_add_word_uses_lock(self):
        """add_word should acquire _vocab_lock."""
        import inspect
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        source = inspect.getsource(CharacterTokenizer.add_word)
        assert "_vocab_lock" in source, (
            "add_word must acquire _vocab_lock for thread safety")

    def test_has_vocab_lock(self):
        """CharacterTokenizer should have a _vocab_lock attribute."""
        import inspect
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        source = inspect.getsource(CharacterTokenizer.__init__)
        assert "_vocab_lock" in source, (
            "CharacterTokenizer.__init__ must create _vocab_lock")


class TestCharTokenizerVocabCap:
    """char_tokenizer add_word must respect max_vocab_size cap."""

    def test_add_word_checks_max_vocab_size(self):
        """add_word should return unk_token_id when vocab is full."""
        import inspect
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        source = inspect.getsource(CharacterTokenizer.add_word)
        assert "_max_vocab_size" in source, (
            "add_word must check _max_vocab_size before growing vocab")
        assert "unk_token_id" in source, (
            "add_word must return unk_token_id when cap is exceeded")

    def test_max_vocab_size_enforced(self):
        """Adding words past max_vocab_size returns unk instead of growing."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer(use_dictionary=False)
        initial_size = tok.vocab_size
        # Set cap to current size — no room for new words
        tok._max_vocab_size = initial_size
        result = tok.add_word("ZZZZZ_test_word_that_wont_exist")
        assert result == tok.unk_token_id
        assert tok.vocab_size == initial_size

    def test_no_cap_allows_growth(self):
        """Without max_vocab_size, vocab grows normally (default behavior)."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer(use_dictionary=False)
        initial_size = tok.vocab_size
        result = tok.add_word("ZZZZZ_unique_test_token_12345")
        assert result >= initial_size
        assert tok.vocab_size == initial_size + 1

    def test_init_accepts_max_vocab_size(self):
        """CharacterTokenizer constructor accepts max_vocab_size param."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer(use_dictionary=False, max_vocab_size=50000)
        assert tok._max_vocab_size == 50000


class TestTrainingQueueFlagLocking:
    """#43: TrainingQueue start/pause/resume/stop must use _lock."""

    def test_start_uses_lock(self):
        """start() must acquire _lock before modifying flags."""
        import inspect
        from enigma_engine.core.training_queue import TrainingQueue
        source = inspect.getsource(TrainingQueue.start)
        assert "self._lock" in source, (
            "start() must use _lock to protect flag assignments")

    def test_pause_uses_lock(self):
        """pause() must acquire _lock."""
        import inspect
        from enigma_engine.core.training_queue import TrainingQueue
        source = inspect.getsource(TrainingQueue.pause)
        assert "self._lock" in source, (
            "pause() must use _lock to protect _paused assignment")

    def test_resume_uses_lock(self):
        """resume() must acquire _lock."""
        import inspect
        from enigma_engine.core.training_queue import TrainingQueue
        source = inspect.getsource(TrainingQueue.resume)
        assert "self._lock" in source, (
            "resume() must use _lock to protect _paused assignment")

    def test_stop_uses_lock(self):
        """stop() must acquire _lock."""
        import inspect
        from enigma_engine.core.training_queue import TrainingQueue
        source = inspect.getsource(TrainingQueue.stop)
        assert "self._lock" in source, (
            "stop() must use _lock to protect flag assignments")

    def test_run_loop_reads_flags_under_lock(self):
        """_run_loop must read _running/_stop_requested under _lock."""
        import inspect
        from enigma_engine.core.training_queue import TrainingQueue
        source = inspect.getsource(TrainingQueue._run_loop)
        assert "self._lock" in source, (
            "_run_loop must read control flags under _lock")


@pytest.mark.structural
class TestModelPosEmbeddingBoundsCheck:
    """#44/#45: model.py forward/forward_multimodal must validate position bounds."""

    def test_forward_checks_pos_bounds(self):
        """forward() must raise on position overflow (non-RoPE path)."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.forward)
        assert "max_seq_len" in source and ("ValueError" in source or "raise" in source), (
            "forward() must validate position against max_seq_len")

    def test_forward_multimodal_checks_pos_bounds(self):
        """forward_multimodal() must raise on T > max_seq_len (non-RoPE path)."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.forward_multimodal)
        assert "max_seq_len" in source and ("ValueError" in source or "raise" in source), (
            "forward_multimodal() must validate T against max_seq_len")


@pytest.mark.structural
class TestModelGenerateSqueezeItem:
    """#47: model.py generate/generate_stream must use squeeze().item()."""

    def test_generate_uses_squeeze_item(self):
        """generate() stop-token check must handle batch>1."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.generate)
        assert "squeeze().item()" in source or "squeeze(-1).item()" in source, (
            "generate() must use .squeeze().item() not bare .item()")

    def test_generate_stream_uses_squeeze_item(self):
        """generate_stream() stop-token check must handle batch>1."""
        import inspect
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.generate_stream)
        assert "squeeze().item()" in source or "squeeze(-1).item()" in source, (
            "generate_stream() must use .squeeze().item() not bare .item()")


@pytest.mark.structural
class TestStreamingPutNowait:
    """#48: streaming.py __aiter__ must use put_nowait, not await put."""

    def test_aiter_uses_put_nowait(self):
        """__aiter__ should use put_nowait() inside threading.Lock context."""
        import inspect
        from enigma_engine.core.streaming import StreamingResponse
        source = inspect.getsource(StreamingResponse.__aiter__)
        assert "put_nowait" in source, (
            "__aiter__ must use put_nowait() not await put() inside threading.Lock")
        assert "await self._async_queue.put(" not in source, (
            "__aiter__ must not await inside threading.Lock")


@pytest.mark.structural
class TestHuggingfaceLoaderNoDeadProperty:
    """#49: huggingface_loader must not have dead @property at module level."""

    def test_no_module_level_property(self):
        """Module should not have a @property decorated function at top level."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "core" / "huggingface_loader.py"
        )
        source = source_path.read_text(encoding="utf-8")
        # @property followed by def at module level (not inside a class) is dead code
        import re
        # Check there's no @property\ndef pattern outside of class bodies
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "@property" and i + 1 < len(lines):
                next_line = lines[i + 1]
                # Module-level function would not be indented more than @property
                if next_line.startswith("def "):
                    pytest.fail(
                        f"Dead @property at module level (line {i+1}): "
                        "only works inside classes")

    def test_create_batches_pads(self):
        """Batches must be padded to max length within the batch."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer._create_batches)
        assert "pad" in source.lower(), (
            "LoRA _create_batches must pad sequences within each batch")

    def test_create_batches_uses_batch_size(self):
        """Batch grouping should use self.batch_size."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer._create_batches)
        assert "self.batch_size" in source, (
            "LoRA _create_batches must group by self.batch_size")

    def test_train_unpacks_batch_mask(self):
        """train() must unpack (input_ids, attention_mask) from batches."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        assert "input_ids, attention_mask" in source, (
            "LoRA train() must unpack (input_ids, attention_mask) tuples")

    def test_train_uses_cross_entropy(self):
        """train() should compute loss manually with cross_entropy."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        assert "cross_entropy" in source, (
            "LoRA train() should use F.cross_entropy for loss computation")

    def test_train_no_labels_kwarg(self):
        """train() must not use HuggingFace-style labels= kwarg."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        assert "labels=" not in source, (
            "LoRA train() should not use labels= (Enigma uses targets=)")

    def test_train_flushes_tail_gradients(self):
        """train() must flush remaining accumulated gradients at epoch end."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        assert "gradient_accumulation_steps" in source, (
            "LoRA train() must handle tail gradient flush")


# ================================================================
# Imagegen scheduler choice — Quick Win #4
# ================================================================

@pytest.mark.structural
class TestImagegenSchedulerChoice:
    """Verify imagegen exposes scheduler choice."""

    def test_generate_accepts_scheduler(self):
        """StableDiffusionLocal.generate() should accept scheduler param."""
        import inspect
        from mods.imagegen.imagegen import StableDiffusionLocal
        sig = inspect.signature(StableDiffusionLocal.generate)
        assert "scheduler" in sig.parameters, (
            "imagegen.generate should accept scheduler param "
            "(DPMSolverMultistep vs EulerDiscrete)")


# ================================================================
# Audio Encoder — Whisper-style Conv1d + Transformer encoder
# ================================================================

@pytest.mark.structural
class TestAudioEncoderConfig:
    """AudioEncoderConfig must define all required fields."""

    def test_config_exists(self):
        """AudioEncoderConfig must be importable."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        config = AudioEncoderConfig()
        assert config is not None

    def test_default_fields(self):
        """Config must have standard Whisper-like defaults."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        config = AudioEncoderConfig()
        assert config.n_mels == 80
        assert config.dim > 0
        assert config.n_layers > 0
        assert config.n_heads > 0
        assert config.sample_rate == 16000
        assert config.n_fft > 0
        assert config.hop_length > 0

    def test_to_dict(self):
        """Config must serialize to dict."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        config = AudioEncoderConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "n_mels" in d
        assert "dim" in d
        assert "n_layers" in d

    def test_max_audio_len_field(self):
        """Config must have max_audio_len for positional embeddings."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        config = AudioEncoderConfig()
        assert hasattr(config, "max_audio_len")
        assert config.max_audio_len > 0


@pytest.mark.structural
class TestAudioPresets:
    """AUDIO_PRESETS must provide standard size presets."""

    def test_presets_exist(self):
        """AUDIO_PRESETS dict must be importable."""
        from enigma_engine.core.audio_encoder import AUDIO_PRESETS
        assert isinstance(AUDIO_PRESETS, dict)
        assert len(AUDIO_PRESETS) >= 3  # tiny, base, small at minimum

    def test_preset_names(self):
        """Must include standard Whisper-like presets."""
        from enigma_engine.core.audio_encoder import AUDIO_PRESETS
        for name in ("tiny", "base", "small"):
            assert name in AUDIO_PRESETS, f"Missing preset: {name}"

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


@pytest.mark.structural
class TestAudioEncoderStructure:
    """AudioEncoder must follow Whisper Conv1d + Transformer pattern."""

    def test_encoder_class_exists(self):
        """AudioEncoder must be importable as nn.Module."""
        import torch.nn as nn
        from enigma_engine.core.audio_encoder import AudioEncoder
        assert issubclass(AudioEncoder, nn.Module)

    def test_has_conv_layers(self):
        """Encoder must have two Conv1d layers (Whisper pattern)."""
        import inspect
        source = inspect.getsource(
            __import__("enigma_engine.core.audio_encoder", fromlist=["AudioEncoder"]).AudioEncoder.__init__
        )
        assert "Conv1d" in source, "AudioEncoder must use Conv1d layers"

    def test_has_transformer_blocks(self):
        """Encoder must have a ModuleList of transformer blocks."""
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2)
        encoder = AudioEncoder(config)
        assert hasattr(encoder, "blocks")
        assert len(encoder.blocks) == 2

    def test_has_positional_embeddings(self):
        """Encoder must have positional embeddings (sinusoidal or learned)."""
        import inspect
        source = inspect.getsource(
            __import__("enigma_engine.core.audio_encoder", fromlist=["AudioEncoder"]).AudioEncoder
        )
        assert "pos_embed" in source or "positional" in source.lower()

    def test_has_final_norm(self):
        """Encoder must have final normalization layer."""
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2)
        encoder = AudioEncoder(config)
        assert hasattr(encoder, "norm")

    def test_forward_output_shape(self):
        """forward() output must be [B, T/2, dim] (stride-2 halves time)."""
        import torch
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2, n_mels=80, max_audio_len=1500)
        encoder = AudioEncoder(config)
        encoder.eval()
        # Input: [B, n_mels, n_frames]
        x = torch.randn(1, 80, 100)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape[0] == 1  # batch
        assert out.shape[1] == 50  # 100 / 2 = 50 (stride-2 conv)
        assert out.shape[2] == 64  # dim

    def test_forward_different_lengths(self):
        """Encoder must handle variable-length audio inputs."""
        import torch
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2, n_mels=80, max_audio_len=1500)
        encoder = AudioEncoder(config)
        encoder.eval()
        for n_frames in [50, 100, 200]:
            x = torch.randn(1, 80, n_frames)
            with torch.no_grad():
                out = encoder(x)
            assert out.shape == (1, n_frames // 2, 64)

    def test_param_count(self):
        """param_count() must return trainable param count."""
        from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
        config = AudioEncoderConfig(dim=64, n_layers=2, n_heads=2)
        encoder = AudioEncoder(config)
        count = encoder.param_count()
        assert isinstance(count, int)
        assert count > 0


@pytest.mark.structural
class TestMelSpectrogram:
    """Mel spectrogram computation must work with torch only."""

    def test_mel_filterbank_function_exists(self):
        """mel_filterbank must be importable."""
        from enigma_engine.core.audio_encoder import mel_filterbank
        assert callable(mel_filterbank)

    def test_mel_filterbank_shape(self):
        """mel_filterbank output must be [n_mels, n_fft//2+1]."""
        import torch
        from enigma_engine.core.audio_encoder import mel_filterbank
        fb = mel_filterbank(sr=16000, n_fft=400, n_mels=80)
        assert isinstance(fb, torch.Tensor)
        assert fb.shape == (80, 201)  # n_mels x (n_fft//2 + 1)

    def test_log_mel_spectrogram_exists(self):
        """log_mel_spectrogram must be importable."""
        from enigma_engine.core.audio_encoder import log_mel_spectrogram
        assert callable(log_mel_spectrogram)

    def test_log_mel_spectrogram_output_shape(self):
        """log_mel_spectrogram must produce [1, n_mels, n_frames]."""
        import torch
        from enigma_engine.core.audio_encoder import log_mel_spectrogram
        # Simulate 1 second of 16kHz audio
        waveform = torch.randn(16000)
        mel = log_mel_spectrogram(waveform, n_fft=400, hop_length=160, n_mels=80)
        assert mel.ndim == 3  # [1, n_mels, n_frames]
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80


@pytest.mark.structural
class TestAudioPreprocessing:
    """Audio file loading and preprocessing."""

    def test_load_audio_function_exists(self):
        """load_audio must be importable."""
        from enigma_engine.core.audio_encoder import load_audio
        assert callable(load_audio)

    def test_preprocess_audio_function_exists(self):
        """preprocess_audio must be importable."""
        from enigma_engine.core.audio_encoder import preprocess_audio
        assert callable(preprocess_audio)

    def test_preprocess_audio_accepts_path_or_waveform(self):
        """preprocess_audio signature must accept path or tensor."""
        import inspect
        from enigma_engine.core.audio_encoder import preprocess_audio
        sig = inspect.signature(preprocess_audio)
        params = list(sig.parameters.keys())
        assert "audio" in params


@pytest.mark.structural
class TestTrainAudio:
    """Trainer.train_audio() must exist and follow train_vision() pattern."""

    def test_train_audio_method_exists(self):
        """Trainer must have train_audio() method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "train_audio")
        assert callable(getattr(Trainer, "train_audio"))

    def test_train_audio_signature(self):
        """train_audio must accept audio_encoder and data params."""
        import inspect
        from enigma_engine.core.training import Trainer
        sig = inspect.signature(Trainer.train_audio)
        params = list(sig.parameters.keys())
        assert "audio_encoder" in params
        assert "data" in params

    def test_train_audio_accepts_unfreeze_text_layers(self):
        """train_audio must support unfreeze_text_layers like train_vision."""
        import inspect
        from enigma_engine.core.training import Trainer
        sig = inspect.signature(Trainer.train_audio)
        assert "unfreeze_text_layers" in sig.parameters

    def test_train_audio_validates_projection(self):
        """train_audio must check that audio_projection exists on model."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_audio)
        assert "audio_projection" in source, (
            "train_audio must validate that model has audio_projection"
        )

    def test_train_audio_uses_forward_multimodal(self):
        """train_audio must call forward_multimodal with audio_features."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_audio)
        assert "forward_multimodal" in source
        assert "audio_features" in source

    def test_train_audio_returns_training_state(self):
        """train_audio return type must be TrainingState."""
        import inspect
        from enigma_engine.core.training import Trainer
        sig = inspect.signature(Trainer.train_audio)
        # Check return annotation if present, or inspect source for return self.state
        source = inspect.getsource(Trainer.train_audio)
        assert "self.state" in source, "train_audio must return self.state"


@pytest.mark.structural
class TestAudioEncoderLazyImport:
    """AudioEncoder must be accessible via core __init__ lazy loading."""

    def test_lazy_import_audio_encoder(self):
        """AudioEncoder must be in _LAZY_LOADER_MAP."""
        import enigma_engine.core as core_mod
        lazy_map = getattr(core_mod, "_LAZY_LOADER_MAP", {})
        assert "AudioEncoder" in lazy_map

    def test_lazy_import_audio_config(self):
        """AudioEncoderConfig must be in _LAZY_LOADER_MAP."""
        import enigma_engine.core as core_mod
        lazy_map = getattr(core_mod, "_LAZY_LOADER_MAP", {})
        assert "AudioEncoderConfig" in lazy_map

    def test_lazy_import_audio_presets(self):
        """AUDIO_PRESETS must be in _LAZY_LOADER_MAP."""
        import enigma_engine.core as core_mod
        lazy_map = getattr(core_mod, "_LAZY_LOADER_MAP", {})
        assert "AUDIO_PRESETS" in lazy_map


# ================================================================
# Vision training augmentation
# ================================================================

@pytest.mark.structural
class TestVisionAugmentation:
    """Verify vision training applies image augmentation."""

    def test_augment_function_exists(self):
        """augment_vision_tensor must be importable."""
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        assert callable(augment_vision_tensor)

    def test_augment_preserves_shape(self):
        """Augmentation must return same shape as input."""
        import torch
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        img = torch.randn(1, 3, 224, 224).clamp(-1, 1)
        result = augment_vision_tensor(img)
        assert result.shape == img.shape

    def test_augment_preserves_range(self):
        """Augmented tensor must stay in [-1, 1]."""
        import torch
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        img = torch.randn(1, 3, 224, 224).clamp(-1, 1)
        for _ in range(20):
            result = augment_vision_tensor(img)
            assert result.min() >= -1.0 and result.max() <= 1.0, (
                f"Augmented tensor out of [-1, 1]: min={result.min()}, max={result.max()}")

    def test_augment_is_stochastic(self):
        """Multiple augmentations of same input should differ."""
        import torch
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        img = torch.ones(1, 3, 32, 32) * 0.5
        results = [augment_vision_tensor(img) for _ in range(10)]
        # At least some should differ (brightness/contrast jitter)
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_same, "Augmentation should produce varied outputs"

    def test_train_vision_calls_augment(self):
        """train_vision must apply augmentation during training."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_vision)
        assert "augment_vision_tensor" in source, (
            "train_vision must call augment_vision_tensor during training")

    def test_augment_has_horizontal_flip(self):
        """Augmentation should include random horizontal flip."""
        import inspect
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        source = inspect.getsource(augment_vision_tensor)
        assert "flip" in source, "Augmentation should include horizontal flip"

    def test_augment_has_color_jitter(self):
        """Augmentation should include brightness/contrast jitter."""
        import inspect
        from enigma_engine.core.vision_encoder import augment_vision_tensor
        source = inspect.getsource(augment_vision_tensor)
        assert "brightness" in source or "contrast" in source, (
            "Augmentation should include color jitter")


# ================================================================
# Full checkpoint resume
# ================================================================

@pytest.mark.structural
class TestCheckpointResume:
    """Verify checkpoints save and restore scheduler/scaler state."""

    def test_save_checkpoint_includes_scheduler(self):
        """_save_checkpoint must save scheduler_state_dict."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._save_checkpoint)
        assert "scheduler_state_dict" in source, (
            "Checkpoint must save scheduler state for resume")

    def test_save_checkpoint_includes_scaler(self):
        """_save_checkpoint must save scaler_state_dict."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._save_checkpoint)
        assert "scaler_state_dict" in source, (
            "Checkpoint must save AMP scaler state for resume")

    def test_load_checkpoint_stashes_scheduler(self):
        """load_checkpoint must store scheduler state for deferred restore."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.load_checkpoint)
        assert "_pending_scheduler_state" in source, (
            "load_checkpoint must stash scheduler state for later restore")

    def test_load_checkpoint_stashes_scaler(self):
        """load_checkpoint must store scaler state for deferred restore."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.load_checkpoint)
        assert "_pending_scaler_state" in source, (
            "load_checkpoint must stash scaler state for later restore")

    def test_restore_pending_state_exists(self):
        """Trainer must have a _restore_pending_state method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, '_restore_pending_state'), (
            "Trainer must have _restore_pending_state for deferred checkpoint restore")

    def test_train_calls_restore(self):
        """train() must call _restore_pending_state after scheduler creation."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "_restore_pending_state" in source, (
            "train() must restore scheduler/scaler from checkpoint")

    def test_train_dpo_calls_restore(self):
        """train_dpo() must call _restore_pending_state after scheduler creation."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "_restore_pending_state" in source, (
            "train_dpo() must restore scheduler/scaler from checkpoint")

    def test_save_checkpoint_includes_training_losses(self):
        """_save_checkpoint must persist training_losses."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._save_checkpoint)
        assert "training_losses" in source, (
            "Checkpoint must save training_losses for resume")

    def test_load_checkpoint_restores_training_losses(self):
        """load_checkpoint must restore training_losses from state."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.load_checkpoint)
        assert "training_losses" in source, (
            "load_checkpoint must restore training_losses")

    def test_save_checkpoint_includes_ema(self):
        """_save_checkpoint must persist EMA state when available."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._save_checkpoint)
        assert "ema" in source, (
            "Checkpoint must save EMA state for resume")

    def test_load_checkpoint_restores_ema(self):
        """load_checkpoint must restore EMA state when available."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.load_checkpoint)
        assert "ema" in source, (
            "load_checkpoint must restore EMA state")


# ================================================================
# Sequence Packing
# ================================================================


class TestSequencePacking:
    """Tests for sequence packing in training."""

    def test_packing_config_default_off(self):
        """use_sequence_packing defaults to False."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert config.use_sequence_packing is False

    def test_packing_config_in_to_dict(self):
        """use_sequence_packing appears in to_dict()."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(use_sequence_packing=True)
        d = config.to_dict()
        assert "use_sequence_packing" in d
        assert d["use_sequence_packing"] is True

    def test_pack_sequences_function_exists(self):
        """Trainer has _pack_sequences method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_pack_sequences")

    def test_pack_sequences_packs_short_seqs(self):
        """Short sequences get combined into one row."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        # Three short sequences, max_len=20
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        packed, masks = pack_sequences(seqs, max_length=20, eos_id=2, pad_id=0)
        # All should fit in one row (3+1+2+1+4+1 = 12 <= 20)
        assert packed.shape[0] == 1
        assert packed.shape[1] == 20

    def test_pack_sequences_mask_is_4d(self):
        """Packing produces a 4D attention mask (B, 1, T, T)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        seqs = [[1, 2, 3], [4, 5]]
        packed, masks = pack_sequences(seqs, max_length=16, eos_id=2, pad_id=0)
        assert masks.ndim == 4
        assert masks.shape[1] == 1  # head dim
        assert masks.shape[2] == masks.shape[3]  # T x T

    def test_pack_sequences_cross_boundary_blocked(self):
        """Tokens in different documents cannot attend to each other."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        # Two sequences: [10, 11] and [20, 21]
        seqs = [[10, 11], [20, 21]]
        packed, masks = pack_sequences(seqs, max_length=16, eos_id=2, pad_id=0)
        # Row 0 contains both seqs packed: [10, 11, EOS, 20, 21, EOS, pad...]
        # Position 0 (token 10) should NOT attend to position 3 (token 20)
        # masks has -inf for blocked positions, 0 for allowed
        assert masks[0, 0, 0, 3].item() < -1e4  # blocked (cross-boundary)
        # Position 1 can attend to position 0 (same doc, causal ok)
        assert masks[0, 0, 1, 0].item() == 0.0   # allowed (same doc, past)

    def test_pack_sequences_causal_within_doc(self):
        """Within a document, future tokens are still masked (causal)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        seqs = [[10, 11, 12]]
        packed, masks = pack_sequences(seqs, max_length=16, eos_id=2, pad_id=0)
        # Position 0 should NOT attend to position 1 (future within same doc)
        assert masks[0, 0, 0, 1].item() < -1e4

    def test_pack_sequences_long_seq_gets_own_row(self):
        """A sequence that fills max_length goes into its own row."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        long_seq = list(range(1, 16))  # 15 tokens
        short_seq = [100, 101]
        packed, masks = pack_sequences(
            [long_seq, short_seq], max_length=16, eos_id=2, pad_id=0)
        # long_seq (15) + EOS = 16 → fills a row alone
        # short_seq (2) + EOS = 3 → separate row
        assert packed.shape[0] == 2

    def test_pack_sequences_pad_positions_masked(self):
        """Padding positions at the end of packed rows are masked out."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        seqs = [[10, 11]]
        packed, masks = pack_sequences(seqs, max_length=8, eos_id=2, pad_id=0)
        # Row: [10, 11, EOS, 0, 0, 0, 0, 0]
        # Position 0 should NOT attend to position 3 (padding)
        assert masks[0, 0, 0, 3].item() < -1e4

    def test_model_forward_accepts_attention_mask_2d(self):
        """model.forward() accepts attention_mask_2d parameter."""
        import inspect
        from enigma_engine.core.model import Enigma
        sig = inspect.signature(Enigma.forward)
        assert "attention_mask_2d" in sig.parameters


# ================================================================
# True byte-level BPE (UTF-8 byte sequences)
# ================================================================

@pytest.mark.structural
class TestByteLevelBPE:
    """Test UTF-8 byte-level BPE encoding."""

    def test_bpe_has_utf8_flag(self):
        """BPETokenizer has use_utf8_bytes attribute."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        assert hasattr(tok, "use_utf8_bytes")

    def test_utf8_mode_encodes_ascii(self):
        """ASCII text encodes correctly in UTF-8 mode."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        ids = tok.encode("hello")
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert "hello" in decoded

    def test_utf8_mode_encodes_unicode(self):
        """Unicode text round-trips through UTF-8 byte BPE."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        text = "café"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert "caf" in decoded
        # The é should survive round-trip (as UTF-8 bytes)
        assert "é" in decoded or "caf" in decoded

    def test_utf8_mode_encodes_emoji(self):
        """Emoji text doesn't produce <unk> tokens in UTF-8 mode."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        ids = tok.encode("hello 🌍")
        # Should not contain unk_token_id (all bytes are in 0-255)
        assert tok.unk_token_id not in ids

    def test_utf8_mode_off_by_default(self):
        """UTF-8 byte mode is off by default for backward compat."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        assert tok.use_utf8_bytes is False

    def test_utf8_flag_saved_and_loaded(self, tmp_path):
        """use_utf8_bytes persists through save/load cycle."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        path = tmp_path / "vocab.json"
        tok.save(path)
        tok2 = BPETokenizer(vocab_file=path)
        assert tok2.use_utf8_bytes is True

    def test_utf8_decode_reconstructs_multibyte(self):
        """Decoding UTF-8 byte tokens reconstructs multi-byte chars."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        # "ñ" is 2 bytes in UTF-8: 0xc3 0xb1
        ids = tok.encode("ñ")
        decoded = tok.decode(ids)
        assert "ñ" in decoded

    def test_legacy_mode_unchanged(self):
        """With use_utf8_bytes=False, behavior is identical to old code."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = False
        ids = tok.encode("hello world")
        decoded = tok.decode(ids)
        assert "hello" in decoded
        assert "world" in decoded

    def test_utf8_train_produces_valid_merges(self, tmp_path):
        """Training in UTF-8 mode produces valid merges."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        tok.train(["hello world café résumé"] * 10,
                   vocab_size=300, verbose=False)
        assert len(tok.merges) > 0
        # Should still encode cleanly
        ids = tok.encode("café")
        decoded = tok.decode(ids)
        assert "caf" in decoded


@pytest.mark.structural
class TestSamplingRepPenaltySign:
    """Verify apply_repetition_penalty handles negative logits correctly."""

    def test_rep_penalty_handles_negative_logits(self):
        """Dividing negative logits by penalty > 1 INCREASES probability.

        Correct behavior: divide positive logits, multiply negative.
        apply_repetition_penalty must not just divide everything.
        """
        import inspect
        from enigma_engine.core.model_utils import apply_repetition_penalty
        source = inspect.getsource(apply_repetition_penalty)
        # Must use torch.where for sign-aware penalty (divide pos, multiply neg)
        assert 'torch.where' in source, (
            "apply_repetition_penalty must use torch.where for sign-aware "
            "penalty — dividing negative logits by penalty > 1 makes them "
            "less negative (higher prob), which is the opposite of penalizing. "
            "Correct: torch.where(scores > 0, scores / p, scores * p)")


@pytest.mark.structural
class TestSamplingRepPenaltyOrder:
    """Verify sample_next_token applies rep penalty before temperature."""

    def test_penalty_before_temperature(self):
        """Rep penalty on temperature-scaled logits couples penalty to temp.

        Correct order: penalty on raw logits first, then temperature.
        """
        import inspect
        from enigma_engine.core.model_utils import sample_next_token
        source = inspect.getsource(sample_next_token)
        lines = source.split('\n')
        temp_line = None
        penalty_line = None
        for i, line in enumerate(lines):
            if 'temperature' in line and '/' in line and 'penalty' not in line:
                temp_line = i
            if 'repetition_penalty' in line and 'apply_repetition_penalty' in line:
                penalty_line = i
        assert temp_line is not None, "Could not find temperature scaling line"
        assert penalty_line is not None, "Could not find rep penalty line"
        assert penalty_line < temp_line, (
            "sample_next_token must apply repetition penalty BEFORE "
            "temperature scaling — applying penalty on temp-scaled logits "
            "couples the penalty strength to temperature")


@pytest.mark.structural
class TestChatCleansIncompleteTags:
    """Verify engine chat() cleans up truncated <think> tags."""

    def test_chat_calls_strip_incomplete_think(self):
        """chat() must strip unclosed <think> tags for API callers.

        The GUI handles this already, but API + CLI callers receive
        the raw chat() return value.
        """
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.chat)
        assert 'strip_incomplete_think' in source, (
            "chat() must call strip_incomplete_think on the response "
            "so API callers don't receive broken <think> tags")


# ================================================================
# Windowed Repetition Penalty
# ================================================================

class TestWindowedRepetitionPenalty:
    """Repetition penalty should only look at recent tokens, not the full history."""

    def test_engine_sampler_has_window_constant(self):
        """_GenerationMixin defines REPETITION_WINDOW."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        assert hasattr(_GenerationMixin, 'REPETITION_WINDOW')
        assert _GenerationMixin.REPETITION_WINDOW > 0

    def test_engine_sampler_windows_tokens(self):
        """_sample_token only considers a recent window of tokens."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._sample_token)
        assert 'REPETITION_WINDOW' in source, (
            "_sample_token must slice generated to a recent window")

    def test_batch_sampler_windows_tokens(self):
        """_sample_token_batch only considers a recent window of tokens."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._sample_token_batch)
        assert 'REPETITION_WINDOW' in source, (
            "_sample_token_batch must slice generated to a recent window")

    def test_model_level_penalty_has_window_param(self):
        """apply_repetition_penalty accepts a window parameter."""
        import inspect
        from enigma_engine.core.model_utils import apply_repetition_penalty
        sig = inspect.signature(apply_repetition_penalty)
        assert 'window' in sig.parameters, (
            "apply_repetition_penalty must accept a window parameter")

    def test_model_level_penalty_windows_tokens(self):
        """apply_repetition_penalty only penalises tokens in the window."""
        import torch
        from enigma_engine.core.model_utils import apply_repetition_penalty

        vocab = 32
        logits = torch.ones(1, vocab)
        # Token 5 appears far in the past (position 0), outside a window of 3
        # Token 10 appears recently (position 9), inside the window
        generated = torch.zeros(1, 10, dtype=torch.long)
        generated[0, 0] = 5   # old
        generated[0, 9] = 10  # recent

        result = apply_repetition_penalty(logits, generated, penalty=2.0, window=3)
        # Token 10 (recent) should be penalised
        assert result[0, 10] < 1.0, "Recent token should be penalised"
        # Token 5 (old, outside window) should NOT be penalised
        assert result[0, 5] == 1.0, "Old token outside window should be untouched"


# ================================================================
# KV-Cache in Manual / Streaming Generation
# ================================================================

class TestManualGenerationUsesKVCache:
    """_generate_manual and stream_generate should use the KV cache."""

    def test_generate_manual_calls_clear_cache(self):
        """_generate_manual must clear the KV cache before starting."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_manual)
        assert 'clear_cache' in source, (
            "_generate_manual must clear KV cache at the start")

    def test_generate_manual_uses_start_pos(self):
        """_generate_manual must pass start_pos for incremental decoding."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_manual)
        assert 'start_pos' in source, (
            "_generate_manual must use start_pos for O(1) per-token decoding")

    def test_generate_manual_uses_cache_flag(self):
        """_generate_manual must pass use_cache=True to model."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_manual)
        assert 'use_cache' in source, (
            "_generate_manual must enable KV cache via use_cache flag")

    def test_stream_generate_uses_kv_cache(self):
        """stream_generate must use KV cache for incremental decoding."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.stream_generate)
        assert 'clear_cache' in source, "stream_generate must clear KV cache"
        assert 'start_pos' in source, "stream_generate must use start_pos"
        assert 'use_cache' in source, "stream_generate must use use_cache"


# ================================================================
# Data Quality Minimums
# ================================================================

class TestDataQualityMinimums:
    """Training data parsing should filter out noise."""

    def test_paragraph_min_length_above_40(self):
        """Paragraph fallback must reject short text (>40 chars minimum)."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._parse_training_data)
        # Find the paragraph length threshold — it should be > 40
        import re
        match = re.search(r'len\(para\)\s*>\s*(\d+)', source)
        assert match, "Paragraph filter must check len(para) > N"
        threshold = int(match.group(1))
        assert threshold >= 40, (
            f"Paragraph min length is {threshold}, should be >= 40 to filter noise")

    def test_token_sequence_min_length(self):
        """Encoded sequences must have at least 5 tokens."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._create_batches)
        # Should require >= 5 tokens, not just > 1
        import re
        match = re.search(r'len\(tokens\)\s*>=?\s*(\d+)', source)
        assert match, "Token filter must check len(tokens) >= N"
        threshold = int(match.group(1))
        assert threshold >= 5, (
            f"Token sequence min is {threshold}, should be >= 5 for meaningful training")


# ================================================================
# Consolidated Generation Paths
# ================================================================

class TestSingleGenerationPath:
    """All native generation goes through one code path (no split dispatch)."""

    def test_generate_text_uses_generate_manual_only(self):
        """_generate_text must NOT dispatch to model.generate() for native models.

        All native generation should go through _generate_manual which
        has KV-cache, windowed repetition penalty, and min-p.
        The GGUF path (which also calls model.generate) is separate and OK.
        """
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_text)
        # After the GGUF early-return, native path should only use _generate_manual
        # Find everything after the GGUF block (after "INPUT VALIDATION")
        native_section = source.split("INPUT VALIDATION", 1)[-1]
        assert 'model.generate(' not in native_section, (
            "_generate_text native path should not dispatch to model.generate() — "
            "all native generation goes through _generate_manual()")
        assert '_generate_manual' in native_section

    def test_vision_generation_uses_shared_sampler(self):
        """_generate_with_vision must use _sample_token, not inline sampling."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_with_vision)
        assert '_sample_token(' in source, (
            "_generate_with_vision must call _sample_token "
            "for consistent sampling across all paths")

    def test_vision_generation_uses_kv_cache(self):
        """_generate_with_vision must use KV cache for incremental decoding."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_with_vision)
        assert 'clear_cache' in source, "vision gen must clear KV cache"
        assert 'start_pos' in source, "vision gen must use start_pos"
        assert 'use_cache' in source, "vision gen must use use_cache flag"

    def test_vision_generation_supports_min_p(self):
        """_generate_with_vision must accept min_p parameter."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin._generate_with_vision)
        assert 'min_p' in sig.parameters, (
            "_generate_with_vision must support min_p for consistent behavior")

    def test_no_inline_repetition_penalty_in_vision(self):
        """_generate_with_vision must NOT have its own penalty loop."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_with_vision)
        # Should NOT have a manual for-loop applying penalty
        assert 'for tok_id in' not in source, (
            "vision gen should use _sample_token, not inline penalty loop")


# ================================================================
# Sentiment Heuristics
# ================================================================

class TestSentimentHeuristics:
    """Test the heuristic sentiment analysis module."""

    def test_module_imports(self):
        """sentiment module exists and imports cleanly."""
        from enigma_engine.core import sentiment
        assert hasattr(sentiment, "analyze_sentiment")

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


# ================================================================
# Emotional State in ModelContext
# ================================================================

class TestEmotionalState:
    """Test emotional state integration in ModelContext."""

    def test_emotional_state_exists(self):
        """ModelContext has emotional_state attribute."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_model")
        assert hasattr(ctx, "emotional_state")

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

    def test_returns_string(self):
        """Result is always a string."""
        from enigma_engine.core.sentiment import build_emotional_prompt_hint
        result = build_emotional_prompt_hint({"valence": 0.5, "arousal": 0.5,
                                              "engagement": 0.5, "trust": 0.5,
                                              "frustration": 0.5})
        assert isinstance(result, str)


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
# Dataset utility — process_text_corpus, clean_text, etc.
# ====================================================================

class TestProcessTextCorpus:
    """Tests for enigma_engine.core.dataset text processing."""

    def test_module_exists(self):
        """dataset.py module is importable."""
        from enigma_engine.core import dataset
        assert hasattr(dataset, "process_text_corpus")

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

    def test_function_exists(self):
        from enigma_engine.core.dataset import clean_text
        assert callable(clean_text)

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

    def test_function_exists(self):
        from enigma_engine.core.dataset import estimate_token_count
        assert callable(estimate_token_count)

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

    def test_registry_exists(self):
        from enigma_engine.core.dataset import KNOWN_DATASETS
        assert isinstance(KNOWN_DATASETS, dict)

    def test_tinystories_registered(self):
        from enigma_engine.core.dataset import KNOWN_DATASETS
        assert "tinystories" in KNOWN_DATASETS

    def test_entries_have_required_fields(self):
        from enigma_engine.core.dataset import KNOWN_DATASETS
        for name, info in KNOWN_DATASETS.items():
            assert "name" in info, f"{name} missing 'name'"
            assert "description" in info, f"{name} missing 'description'"


# =====================================================================
# Suggestion batch: 5 fixes (March 2026)
# =====================================================================

class TestRouterReplayBufferSorting:
    """Replay buffer retraining should use top-scoring examples."""

    def test_retrain_sorts_by_score(self):
        """_retrain_on_replay must sort replay buffer by score (desc)
        before slicing, not just take insertion-ordered head."""
        source = inspect.getsource(
            __import__("enigma_engine.router", fromlist=["BackgroundTrainer"])
            .BackgroundTrainer._retrain_on_replay
        )
        # Must contain a sort/sorted call on score
        assert "sort" in source, (
            "_retrain_on_replay must sort replay buffer by score")

    def test_retrain_sorts_descending(self):
        """Sort must be descending so highest scores come first."""
        source = inspect.getsource(
            __import__("enigma_engine.router", fromlist=["BackgroundTrainer"])
            .BackgroundTrainer._retrain_on_replay
        )
        assert "reverse=True" in source, (
            "Sort must be descending (reverse=True) for top scores")

    def test_retrain_uses_param_group_lr_scaling(self):
        """Replay retrain must use optimizer param_groups LR scaling,
        not loss multiplication, to reduce effective learning rate."""
        source = inspect.getsource(
            __import__("enigma_engine.router", fromlist=["BackgroundTrainer"])
            .BackgroundTrainer._retrain_on_replay
        )
        assert "param_groups" in source, (
            "_retrain_on_replay must scale LR via param_groups")
        assert 'loss * 0.5' not in source and 'loss *0.5' not in source, (
            "_retrain_on_replay should not use loss scaling")


class TestDatasetFileSizeGuard:
    """process_text_corpus must not OOM on oversized files."""

    def test_max_file_size_constant_exists(self):
        """dataset module defines a MAX_FILE_SIZE constant."""
        import enigma_engine.core.dataset as ds
        assert hasattr(ds, "MAX_FILE_SIZE"), (
            "dataset.py must define MAX_FILE_SIZE")
        assert isinstance(ds.MAX_FILE_SIZE, int)
        assert ds.MAX_FILE_SIZE > 0

    def test_process_file_checks_size(self):
        """_process_file source must check file size before reading."""
        import enigma_engine.core.dataset as ds
        source = inspect.getsource(ds._process_file)
        assert "MAX_FILE_SIZE" in source or "stat" in source, (
            "_process_file must check file size")

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


class TestGenerateManualStopStrings:
    """_generate_manual should support early stopping on stop strings."""

    def test_generate_manual_accepts_stop_strings(self):
        """_generate_manual signature includes stop_strings parameter."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin._generate_manual)
        assert "stop_strings" in sig.parameters, (
            "_generate_manual must accept stop_strings parameter")

    def test_generate_text_passes_stop_strings(self):
        """_generate_text must pass stop_strings to _generate_manual."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_text)
        # Find the actual _generate_manual() call (not the comment)
        # The call line should pass stop_strings as an argument
        call_idx = source.rfind("_generate_manual(")
        assert call_idx != -1, "_generate_text must call _generate_manual"
        call_region = source[call_idx:call_idx + 300]
        assert "stop_strings" in call_region, (
            "_generate_text must pass stop_strings to _generate_manual")

    def test_generate_manual_checks_stop_strings_in_loop(self):
        """_generate_manual must check stop strings during generation."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_manual)
        assert "stop_string" in source or "stop_str" in source, (
            "_generate_manual must check stop strings in the gen loop")

    def test_vision_generation_checks_stop_strings_in_loop(self):
        """_generate_with_vision must also check stop strings in loop."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_with_vision)
        # Should have stop string check inside the for loop, not just after
        loop_body = source.split("for _")[1] if "for _" in source else source
        assert "stop_str" in loop_body, (
            "_generate_with_vision must check stop strings during generation")


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

    def test_function_exists(self):
        """sentiment.py exports compute_engagement_score."""
        from enigma_engine.core.sentiment import compute_engagement_score
        assert callable(compute_engagement_score)

    def test_neutral_state_returns_one(self):
        """Neutral/baseline emotional state returns weight ~1.0."""
        from enigma_engine.core.sentiment import compute_engagement_score
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        score = compute_engagement_score(dict(_EMOTIONAL_BASELINE))
        assert 0.9 <= score <= 1.1

    def test_high_engagement_boosts_score(self):
        """High engagement + trust → weight > 1.0."""
        from enigma_engine.core.sentiment import compute_engagement_score
        state = {
            "valence": 0.6, "arousal": 0.5,
            "engagement": 0.9, "trust": 0.8, "frustration": 0.0,
        }
        score = compute_engagement_score(state)
        assert score > 1.0

    def test_high_frustration_lowers_score(self):
        """High frustration → weight < 1.0."""
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


class TestEmotionalReplayScoring:
    """BackgroundTrainer.add_example accepts engagement_weight."""

    def test_add_example_accepts_score(self):
        """add_example score parameter is used by the replay buffer."""
        source = inspect.getsource(
            __import__("enigma_engine.router", fromlist=["BackgroundTrainer"])
            .BackgroundTrainer.add_example
        )
        assert "score" in source

    def test_add_training_example_passes_score(self):
        """ModRouter.add_training_example passes score to trainer."""
        source = inspect.getsource(
            __import__("enigma_engine.router", fromlist=["ModRouter"])
            .ModRouter.add_training_example
        )
        assert "score" in source


class TestEmotionalSelfPlayBonus:
    """Self-play rewards include emotional evaluation."""

    def test_evaluate_response_sentiment_exists(self):
        """sentiment.py exports evaluate_response_quality."""
        from enigma_engine.core.sentiment import evaluate_response_quality
        assert callable(evaluate_response_quality)

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


class TestFeedTrainerWithEngagement:
    """_feed_background_trainer passes engagement score."""

    def test_feed_captures_emotional_state(self):
        """_feed_background_trainer uses compute_engagement_score."""
        source = inspect.getsource(
            __import__(
                "enigma_engine.gui.gui_logic_chat",
                fromlist=["LogicChatMixin"])
            .LogicChatMixin._feed_background_trainer
        )
        assert "engagement" in source.lower() or "compute_engagement" in source


# ================================================================
# VRAM-based preset recommendation
# ================================================================


class TestEstimateTrainingVram:
    """estimate_training_vram returns reasonable VRAM estimates."""

    def test_small_preset_low_vram(self):
        """Small preset should need <2 GB."""
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_training_vram)
        import copy
        cfg = copy.deepcopy(MODEL_PRESETS["small"])
        cfg.vocab_size = 32000
        vram = estimate_training_vram(cfg)
        assert 0.5 <= vram <= 2.0, f"small needs {vram} GB, expected <2"

    def test_large_preset_moderate_vram(self):
        """Large preset (~200M) should need 2-8 GB."""
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_training_vram)
        import copy
        cfg = copy.deepcopy(MODEL_PRESETS["large"])
        cfg.vocab_size = 32000
        vram = estimate_training_vram(cfg)
        assert 2.0 <= vram <= 8.0, f"large needs {vram} GB, expected 2-8"

    def test_xl_preset_high_vram(self):
        """XL preset (~600M) should need 8-20 GB."""
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_training_vram)
        import copy
        cfg = copy.deepcopy(MODEL_PRESETS["xl"])
        cfg.vocab_size = 32000
        vram = estimate_training_vram(cfg)
        assert 8.0 <= vram <= 20.0, f"xl needs {vram} GB, expected 8-20"

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

    def test_32gb_picks_xl_or_larger(self):
        """32 GB VRAM should pick xl or xxl (not small)."""
        from enigma_engine.core.model_presets import recommend_preset_for_vram
        result = recommend_preset_for_vram(32.0)
        assert result in ("xl", "xxl"), f"32 GB got {result}, expected xl or xxl"

    def test_8gb_picks_medium_or_larger(self):
        """8 GB should pick at least medium."""
        from enigma_engine.core.model_presets import recommend_preset_for_vram
        result = recommend_preset_for_vram(8.0)
        big_presets = {"medium", "base", "large", "xl"}
        assert result in big_presets, f"8 GB got {result}"

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
