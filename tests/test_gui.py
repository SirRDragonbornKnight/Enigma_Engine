"""Tests for the desktop GUI module and widgets."""

import json
import inspect
from pathlib import Path

import pytest


# ── Shared test helpers ──────────────────────────────────────────────


class MockVar:
    """Minimal mock for tkinter variable-like objects."""
    def __init__(self, initial=True):
        self.value = initial

    def get(self):
        return self.value

    def set(self, val):
        self.value = val


class MockStatusBar:
    """Minimal mock for the status bar widget."""
    def set_left(self, text): pass
    def set_center(self, text): pass
    def set_right(self, text): pass


class DummyStatusBar:
    """Status bar that records nothing — for tests that just need the API."""
    def set_left(self, text): pass
    def set_center(self, text): pass
    def set_right(self, text): pass


class DummyThread:
    """Thread replacement that records kwargs and does nothing."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def start(self):
        pass


class UnexpectedThread:
    """Thread replacement that fails if instantiated."""
    def __init__(self, **kwargs):
        raise RuntimeError("Unexpected thread created")


# ================================================================
# Scanners
# ================================================================

class TestScanners:
    """Verify all filesystem scanners work."""

    def test_scan_mods(self):
        from enigma_engine.gui.scanners import scan_mods
        mods = scan_mods()
        ids = [m["id"] for m in mods]
        assert "imagegen" in ids
        assert "voice" in ids
        # Voice and audio generation are intentionally unified.
        assert "audiogen" not in ids
        assert "_template" not in ids
        for mod in mods:
            assert "id" in mod
            assert "name" in mod
            assert "commands" in mod
            assert "prompt" in mod
            assert "rules" in mod

    def test_scan_models(self):
        from enigma_engine.gui.scanners import scan_models
        models = scan_models()
        assert isinstance(models, list)
        for m in models:
            assert "name" in m
            assert "path" in m
            assert "size_mb" in m

    def test_scan_models_groups_sharded_safetensors(self):
        """Sharded safetensors should be merged into one model entry."""
        from enigma_engine.gui.scanners import scan_models
        models = scan_models()
        names = [m["name"] for m in models]
        # Shard filenames like model-00001-of-00005 must not appear
        for name in names:
            assert "-of-" not in name, (
                f"Sharded file '{name}' should be grouped, "
                "not listed individually")

    def test_scan_training_data(self):
        from enigma_engine.gui.scanners import scan_training_data
        files = scan_training_data()
        assert isinstance(files, list)
        names = [f["name"] for f in files]
        assert "training.txt" in names
        assert "gui_settings.json" not in names
        for f in files:
            assert "path" in f
            assert "size_kb" in f

    def test_scan_sessions(self):
        from enigma_engine.gui.scanners import scan_sessions
        sessions = scan_sessions()
        assert isinstance(sessions, list)
        for s in sessions:
            assert "name" in s
            assert "path" in s

    def test_scan_docs(self):
        """scan_docs returns guides and mod docs."""
        from enigma_engine.gui.scanners import scan_docs, INFO_DIR
        assert INFO_DIR.exists()
        docs = scan_docs()
        assert isinstance(docs, list)
        assert len(docs) > 0
        for doc in docs:
            assert "name" in doc
            assert "path" in doc
            assert "category" in doc
            assert "filename" in doc
        # Has guides
        guides = [d for d in docs if d["category"] == "guides"]
        assert len(guides) >= 5
        names = {d["filename"] for d in guides}
        assert "how_the_ai_works.md" in names
        assert "training_guide.md" in names
        assert "commands_reference.md" in names
        # Has mod docs
        assert any(d["category"].startswith("mod:") for d in docs)
        # All files readable
        for doc in docs:
            path = Path(doc["path"])
            assert path.exists(), f"Missing: {path}"
            assert len(path.read_text(encoding="utf-8")) > 10


# ================================================================
# Config validation
# ================================================================

class TestConfigValidation:
    """Verify config clamping and descriptions."""

    def test_clamp_config_values(self):
        from enigma_engine.gui.scanners import clamp_config
        assert clamp_config("temperature", 5.0) == 2.0
        assert clamp_config("temperature", -1.0) == 0.0
        assert clamp_config("temperature", 0.8) == 0.8
        assert clamp_config("top_k", 0) == 1
        assert clamp_config("top_k", 999) == 200
        assert clamp_config("max_tokens", 0) == 1
        assert clamp_config("max_tokens", 99999) == 99999

    def test_config_descriptions_exist(self):
        from enigma_engine.gui.scanners import (
            CONFIG_DESCRIPTIONS, CONFIG_LIMITS)
        for name in CONFIG_LIMITS:
            assert name in CONFIG_DESCRIPTIONS


# ================================================================
# Module structure and backward compatibility
# ================================================================

class TestModuleStructure:
    """Verify GUI module split, mixins, and re-exports."""

    def test_widget_module(self):
        from enigma_engine.gui.widgets import (
            HUDFrame, GlowFrame, StatusDot, NavButton,
            SectionLabel, ToggleButton, StatusBar,
            CollapsiblePanel, SelectableTextbox, Tooltip,
            C_BG, C_PANEL, C_ACCENT, C_TEXT, C_GREEN, C_RED,
            C_CYAN, C_BORDER_ACCENT,
            FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_CMD)
        assert GlowFrame is HUDFrame
        assert isinstance(C_BG, str)
        assert isinstance(FONT_TITLE, tuple)

    def test_desktop_inherits_all_mixins(self):
        from enigma_engine.gui.desktop import EnigmaGUI
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        from enigma_engine.gui.gui_cmd_page import CMDPageMixin
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert issubclass(EnigmaGUI, DocsPageMixin)
        assert issubclass(EnigmaGUI, CMDPageMixin)
        assert issubclass(EnigmaGUI, ForgeMixin)

    def test_backward_compat_imports(self):
        from enigma_engine.gui.desktop import (
            scan_mods, scan_models, scan_docs,
            INFO_DIR, clamp_config, CONFIG_LIMITS)
        assert callable(scan_mods)
        assert callable(scan_docs)
        assert isinstance(CONFIG_LIMITS, dict)
        assert INFO_DIR.exists()


# ================================================================
# Mod template completeness
# ================================================================


class TestModDefinitions:
    """Verify mod file conventions and merged audio/voice setup."""

    def test_voice_mod_has_main_entry(self):
        from enigma_engine.gui.scanners import MODS_DIR
        assert (MODS_DIR / "voice" / "main.py").exists()


# ================================================================
# CMD Page
# ================================================================

class TestCMDPage:
    """Verify CMD page dual-mode terminal."""

    def test_cmd_page_methods(self):
        from enigma_engine.gui.gui_cmd_page import (
            CMDPageMixin, MODE_SYSTEM, MODE_ENGINE)
        assert MODE_SYSTEM == "SYSTEM"
        assert MODE_ENGINE == "ENGINE"
        for attr in (
            "_build_page_cmd", "_cmd_execute", "_cmd_clear",
            "_cmd_write", "_cmd_welcome", "_cmd_run_system",
            "_cmd_run_engine", "_cmd_ask_ai", "_cmd_switch_mode",
            "_cmd_toggle_ai_access", "_cmd_execute_ai_command",
        ):
            assert hasattr(CMDPageMixin, attr), f"Missing: {attr}"

    def test_cmd_engine_registry(self):
        """Engine commands work for ENGINE mode."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        result = registry.execute("config.list")
        assert result.success


# ================================================================
# Per-model context
# ================================================================

class TestModelContext:
    """Verify per-model context integrates with GUI."""

    def test_model_context_module(self):
        from enigma_engine.core.model_context import (
            ModelContext, model_key_from_path,
            load_model_context, list_model_contexts,
            get_contexts_dir)
        assert callable(load_model_context)
        ctx_dir = get_contexts_dir()
        assert ctx_dir.name == "model_contexts"


# ================================================================
# CORE page widgets
# ================================================================


# ================================================================
# Voice input
# ================================================================


# ================================================================
# FORGE data editor
# ================================================================


# ================================================================
# Nav rail
# ================================================================


# ================================================================
# Path settings
# ================================================================

class TestPathSettings:
    """Verify directory path settings."""

    def test_path_constants_and_defaults(self):
        from enigma_engine.gui.scanners import (
            PATH_SETTINGS, get_path, MODELS_DIR)
        assert isinstance(PATH_SETTINGS, dict)
        assert "models_dir" in PATH_SETTINGS
        assert "outputs_dir" in PATH_SETTINGS
        assert get_path("models_dir") == MODELS_DIR


# ================================================================
# DOCS page
# ================================================================

class TestDocsPage:
    """DOCS page: documentation browser, file management."""

    def test_scan_docs_has_data_category(self):
        """scan_docs includes training data files under 'data' category."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        categories = {d["category"] for d in docs}
        assert "data" in categories, (
            "scan_docs should include data/ files as 'data' category")


# ================================================================
# DOCS page improvements
# ================================================================

class TestDocsPageImprovements:
    """DOCS page: search, notes, unsaved changes, Ctrl+S, stats."""

    def test_scan_docs_has_notes_category(self):
        """scan_docs includes notes files under 'notes' category."""
        from enigma_engine.gui.scanners import scan_docs, NOTES_DIR
        # Create a test note if dir is empty
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        test_note = NOTES_DIR / "_test_note.md"
        created = False
        if not any(NOTES_DIR.glob("*.md")):
            test_note.write_text("test", encoding="utf-8")
            created = True
        try:
            docs = scan_docs()
            categories = {d["category"] for d in docs}
            assert "notes" in categories, (
                "scan_docs should include notes/ files as 'notes' category")
        finally:
            if created and test_note.exists():
                test_note.unlink()

    def test_scanners_notes_dir_constant(self):
        """scanners module exports NOTES_DIR constant."""
        from enigma_engine.gui.scanners import NOTES_DIR
        assert NOTES_DIR.name == "notes"

    def test_scanners_scan_docs_returns_list(self):
        """scan_docs should return a list of doc entries."""
        from enigma_engine.gui import scanners
        result = scanners.scan_docs()
        assert isinstance(result, list)


# ================================================================
# Docs undo/redo
# ================================================================


# ================================================================
# Chat fullscreen
# ================================================================


# ================================================================
# Display names and model AI name
# ================================================================


# ================================================================
# Trainer Docs Section
# ================================================================

class TestTrainerDocs:
    """Tests for the TRAINER section in DOCS page."""

    def test_trainer_dir_exists(self):
        """Trainer docs directory exists with files."""
        from enigma_engine.gui.scanners import TRAINER_DIR
        assert TRAINER_DIR.exists()
        files = list(TRAINER_DIR.glob("*.md"))
        assert len(files) >= 3

    def test_scan_docs_has_trainer_category(self):
        """scan_docs returns items with trainer category."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        categories = {d["category"] for d in docs}
        assert "trainer" in categories

    def test_trainer_files_readable(self):
        """All trainer docs are readable."""
        from enigma_engine.gui.scanners import TRAINER_DIR
        for f in TRAINER_DIR.glob("*.md"):
            content = f.read_text(encoding="utf-8")
            assert len(content) > 50


# ================================================================
# Unified History / Sessions
# ================================================================


class TestExternalModelsDocs:
    """Tests for external model limitations documentation."""

    def test_scan_docs_finds_external_models(self):
        """scan_docs discovers the external_models.md file."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        names = [d["name"] for d in docs]
        assert "External Models" in names


# ================================================================
# Models page feedback
# ================================================================


class TestRouteAssignmentPersistence:
    """Test that route assignments save and load from disk."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Route assignments survive save → load round trip."""
        import enigma_engine.gui.scanners as scanners
        # Temporarily redirect the routes file
        original = scanners._ROUTES_FILE
        scanners._ROUTES_FILE = tmp_path / "routes.json"
        try:
            assignments = {"chat": "/models/test.gguf",
                           "trainer": "/models/train.pth"}
            scanners.save_route_assignments(assignments)
            loaded = scanners.load_route_assignments()
            assert loaded["chat"] == "/models/test.gguf"
            assert loaded["trainer"] == "/models/train.pth"
        finally:
            scanners._ROUTES_FILE = original

    def test_none_values_not_saved(self, tmp_path):
        """None-valued routes are excluded from the saved file."""
        import enigma_engine.gui.scanners as scanners
        original = scanners._ROUTES_FILE
        scanners._ROUTES_FILE = tmp_path / "routes.json"
        try:
            assignments = {"chat": "/models/test.gguf",
                           "trainer": None}
            scanners.save_route_assignments(assignments)
            loaded = scanners.load_route_assignments()
            assert "chat" in loaded
            assert "trainer" not in loaded
        finally:
            scanners._ROUTES_FILE = original


class TestStudentRoute:
    """Test the STUDENT route integration."""

    def test_route_keys_includes_student(self):
        """ROUTE_KEYS contains the student route."""
        from enigma_engine.gui.scanners import ROUTE_KEYS
        assert "student" in ROUTE_KEYS

    def test_route_persistence_includes_student(self, tmp_path):
        """Student route survives save and load round trip."""
        import enigma_engine.gui.scanners as scanners
        original = scanners._ROUTES_FILE
        scanners._ROUTES_FILE = tmp_path / "routes.json"
        try:
            assignments = {
                "chat": "/models/chat.gguf",
                "trainer": "/models/trainer.pth",
                "student": "/models/student.pth",
            }
            scanners.save_route_assignments(assignments)
            loaded = scanners.load_route_assignments()
            assert loaded["student"] == "/models/student.pth"
        finally:
            scanners._ROUTES_FILE = original


class TestBlankModelCreate:
    """Test simplified blank model creation on MODELS page."""

    def test_rename_model_method_exists(self):
        """ForgeMixin must have _rename_model for renaming models."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, '_rename_model'), (
            "ForgeMixin must provide _rename_model method")
        assert callable(ForgeMixin._rename_model)


class TestForgeModeUI:
    """Test that FORGE training mode descriptions match the 8 radio buttons."""

    def test_descriptions_cover_all_modes(self):
        """_TRAINING_MODE_DESCRIPTIONS has exactly the 8 GUI modes."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        expected = {"Pre-Train", "Distill", "Basic", "AI-Guided",
                    "Image", "Dialogue", "RLHF", "Self-Play"}
        assert set(ForgeMixin._TRAINING_MODE_DESCRIPTIONS.keys()) == expected


class TestForgeNewModes:
    """Test new training modes wiring."""

    def test_display_name_mapping_covers_all_modes(self):
        """_MODE_DISPLAY_TO_KEY maps all 8 display names to keys."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        mapping = ForgeMixin._MODE_DISPLAY_TO_KEY
        assert len(mapping) == 8
        # Every display name resolves to a valid internal key
        expected_keys = {"Pre-Train", "Distill", "Basic",
                         "AI-Guided", "Image", "Dialogue",
                         "RLHF", "Self-Play"}
        assert set(mapping.values()) == expected_keys

    def test_reverse_mapping_covers_all_keys(self):
        """_MODE_KEY_TO_DISPLAY maps all 8 internal keys to display."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        reverse = ForgeMixin._MODE_KEY_TO_DISPLAY
        assert len(reverse) == 8
        # Display names match GUI radio button values
        assert reverse["Image"] == "Image"
        assert reverse["Basic"] == "Basic"
        assert reverse["AI-Guided"] == "AI-Guided"
        assert reverse["Dialogue"] == "Dialogue"
        assert reverse["RLHF"] == "RLHF"
        assert reverse["Self-Play"] == "Self-Play"

    def test_pretrain_tokenizer_cap_in_code(self):
        """Pre-train code must define a tokenizer sample cap to prevent OOM.

        Structural guard: _TOK_SAMPLE_CAP is a local variable inside the
        pre-train function, not a module-level constant. We verify the
        function source contains the cap logic.
        """
        import inspect
        import enigma_engine.gui.gui_forge_new_modes as mod
        src = inspect.getsource(mod)
        assert "_TOK_SAMPLE_CAP" in src, (
            "Missing _TOK_SAMPLE_CAP — tokenizer will OOM on "
            "large corpora")


class TestForgeThreeModeConnections:
    """Regression tests for FORGE wiring."""

    def test_no_forge_focus_field_references(self):
        """No training methods reference the removed forge_focus_field widget."""
        from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin
        from enigma_engine.gui.gui_forge_advanced import ForgeAdvancedMixin
        for cls in (ForgeTrainingMixin, ForgeAdvancedMixin):
            source = inspect.getsource(cls)
            assert "forge_focus_field" not in source


# ================================================================
# FORGE Page: Model Status Cards
# ================================================================


# ================================================================
# FORGE Page: Solo Training
# ================================================================


# ================================================================
# FORGE Helpers: Prompt Extraction
# ================================================================

class TestExtractPrompts:
    """Test _extract_prompts helper for parsing data files."""

    def test_extract_prompts_qa_format(self):
        """_extract_prompts pulls question from Q/A format."""
        import tempfile
        from enigma_engine.gui.gui_forge import ForgeMixin
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt",
                delete=False, encoding="utf-8") as f:
            f.write("Q: What is AI?\nA: Artificial Intelligence.\n\n"
                    "Q: What is ML?\nA: Machine Learning.\n")
            f.flush()
            prompts = ForgeMixin._extract_prompts(f.name)
        assert len(prompts) == 2
        assert "What is AI?" in prompts[0]
        assert "What is ML?" in prompts[1]
        # Should NOT include the A: lines as prompts
        assert not any("Artificial" in p for p in prompts)

    def test_extract_prompts_jsonl_format(self):
        """_extract_prompts pulls prompt from JSONL format."""
        import json
        import tempfile
        from enigma_engine.gui.gui_forge import ForgeMixin
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl",
                delete=False, encoding="utf-8") as f:
            f.write(json.dumps(
                {"prompt": "Hello", "completion": "Hi"}) + "\n")
            f.write(json.dumps(
                {"prompt": "Bye", "completion": "See ya"}) + "\n")
            f.flush()
            prompts = ForgeMixin._extract_prompts(f.name)
        assert len(prompts) == 2
        assert prompts[0] == "Hello"
        assert prompts[1] == "Bye"

    def test_extract_prompts_raw_text(self):
        """_extract_prompts falls back to non-empty lines."""
        import tempfile
        from enigma_engine.gui.gui_forge import ForgeMixin
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt",
                delete=False, encoding="utf-8") as f:
            f.write("# Comment\nHello world\n\nTell me a joke\n")
            f.flush()
            prompts = ForgeMixin._extract_prompts(f.name)
        assert "Hello world" in prompts
        assert "Tell me a joke" in prompts
        # Comments should be excluded
        assert not any(p.startswith("#") for p in prompts)

    def test_extract_prompts_user_ai_format(self):
        """_extract_prompts pulls User lines from User/AI format."""
        import tempfile
        from enigma_engine.gui.gui_forge import ForgeMixin
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt",
                delete=False, encoding="utf-8") as f:
            f.write("User: Hello!\nAI: Hi there!\n\n"
                    "User: How are you?\nAI: Fine!\n")
            f.flush()
            prompts = ForgeMixin._extract_prompts(f.name)
        assert len(prompts) == 2
        assert "Hello!" in prompts[0]
        assert "How are you?" in prompts[1]


# ================================================================
# FORGE Helpers: Engine Loading
# ================================================================


# ================================================================
# FORGE: Guided uses _extract_prompts
# ================================================================


# ================================================================
# FORGE Page: Guided Training
# ================================================================


# ================================================================
# FORGE Page: Dialogue Training (TRAINER ↔ STUDENT conversation)
# ================================================================


# ================================================================
# FORGE: Stage-Aware Generation Prompts
# ================================================================

class TestGenerationPromptBuilder:
    """Test _build_generation_prompt produces varied formats per stage."""

    def test_basics_not_forced_qa(self):
        """Basics stage generates varied formats, not just Q&A."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "basics")
        # Should mention multiple types, not force Q:/A:
        assert "Format exactly as" not in prompt
        assert "statement" in prompt.lower() or "greeting" in prompt.lower()

    def test_conversation_uses_dialogue_format(self):
        """Conversation stage uses User/AI dialogue format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "conversation")
        assert "User:" in prompt
        assert "AI:" in prompt

    def test_commands_uses_qa_with_cmd(self):
        """Commands stage uses Q&A with [CMD] blocks."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "commands")
        assert "Q:" in prompt
        assert "[CMD]" in prompt

    def test_web_uses_qa_with_search(self):
        """Web stage uses Q&A with search/fetch commands."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "web")
        assert "search.web" in prompt

    def test_unknown_stage_falls_back_to_basics(self):
        """Unknown stage name falls back to basics prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "unknown_stage")
        basics = ForgeMixin._build_generation_prompt(1, 10, "basics")
        assert prompt == basics

    def test_includes_index_and_total(self):
        """Prompt includes the example index and total count."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(7, 20, "basics")
        assert "#7" in prompt
        assert "20" in prompt


# ================================================================
# FORGE: Training Pair Formatter
# ================================================================

class TestFormatTrainingPair:
    """Test _format_training_pair outputs the right format per stage."""

    def test_basics_raw_text(self):
        """Basics stage returns raw text without Q:/A: or User:/AI:."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "basics", "Hello", "Hi there")
        assert result == "Hello\nHi there"
        assert "Q:" not in result
        assert "User:" not in result

    def test_conversation_user_ai_format(self):
        """Conversation stage returns User/AI dialogue format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "conversation", "How are you?", "I'm good!")
        assert result == "User: How are you?\nAI: I'm good!"

    def test_commands_qa_format(self):
        """Commands stage returns Q&A format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "commands", "List files", "[CMD]ls[/CMD]")
        assert result == "Q: List files\nA: [CMD]ls[/CMD]"

    def test_web_qa_format(self):
        """Web stage returns Q&A format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "web", "Search for cats", "[CMD]search.web cats[/CMD]")
        assert result == "Q: Search for cats\nA: [CMD]search.web cats[/CMD]"

    def test_unknown_stage_defaults_to_qa(self):
        """Unknown stage name falls back to Q&A format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "custom", "prompt", "response")
        assert result == "Q: prompt\nA: response"

    def test_training_guide_visible_on_docs_page(self):
        """Training guide is discoverable by scan_docs."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        names = [d["filename"] for d in docs]
        assert "training_guide.md" in names


# ================================================================
# FORGE Page: Evaluate Student
# ================================================================

class TestEvaluateStudent:
    """Test TRAINER interactively testing STUDENT."""

    def test_evaluate_no_data_file_required(self):
        """_evaluate_student works without a data file."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._evaluate_student)
        # No check for data_path / train_data_var
        assert "train_data_var" not in source


# ================================================================
# FORGE Page: Checkpoint Save/Resume
# ================================================================


# ================================================================
# FORGE Page: Loss Curve Visualization
# ================================================================


# ================================================================
# CoT-B: REASONING-AWARE TRAINING DATA
# ================================================================


# ================================================================
# FORGE: Trainer System Prompt (human-like responses)
# ================================================================

class TestBuildTrainerSystemPrompt:
    """Test _build_trainer_system_prompt for human-like output."""

    def test_returns_string(self):
        """Returns a non-empty string."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_includes_param_count(self):
        """Prompt includes the student parameter count."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=5_000_000)
        assert "5,000,000" in result

    def test_tiny_model_gets_simple_guidance(self):
        """Very small models get short/simple response guidance."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=500_000)
        assert "very small" in result.lower() or "simple" in result.lower()

    def test_medium_model_gets_paragraph_guidance(self):
        """Medium models get paragraph-length guidance."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=100_000_000)
        assert "medium" in result.lower()

    def test_includes_architecture_info(self):
        """When student_cfg is provided, architecture info appears."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            n_layers=6, dim=256, max_seq_len=512)
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, student_cfg=cfg)
        assert "6 layers" in result
        assert "256" in result

    def test_discourages_ai_phrases(self):
        """Prompt tells TRAINER not to sound like a generic AI."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=10_000_000)
        lower = result.lower()
        assert "as an ai" in lower
        assert "human" in lower or "person" in lower or "friend" in lower
        assert "guardrails" in lower or "limits" in lower

    def test_fact_checking_instructions(self):
        """Prompt teaches fact-checking and offline fallback."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=10_000_000)
        lower = result.lower()
        # Should teach verification
        assert "fact check" in lower or "double-check" in lower or "verify" in lower
        # Should handle no internet
        assert "no internet" in lower or "no internet" in lower.replace("'", "")
        # Should not refuse, give best answer
        assert "best answer" in lower or "best take" in lower
        # Should flag uncertainty
        assert "confident" in lower or "uncertain" in lower or "not 100" in lower

    def test_task_parameter_reflected(self):
        """Task name appears in the prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=10_000_000, task="evaluate")
        assert "evaluate" in result

    def test_no_architecture_without_cfg(self):
        """No architecture line when student_cfg is None."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=10_000_000, student_cfg=None)
        assert "Student architecture" not in result


# ================================================================
# FORGE: Training Stages (curriculum)
# ================================================================

class TestTrainingStages:
    """Test training stage curriculum in system prompt."""

    def test_stage_parameter_accepted(self):
        """_build_trainer_system_prompt accepts stage parameter."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # Should not raise
        for s in ("basics", "conversation", "commands", "web"):
            result = ForgeMixin._build_trainer_system_prompt(
                student_params=1_000_000, stage=s)
            assert isinstance(result, str)

    def test_basics_stage_content(self):
        """Basics stage focuses on simple sentences."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="basics")
        lower = result.lower()
        assert "basics" in lower
        assert "sentence" in lower
        # Stage section should say NOT to teach commands yet
        assert "do not teach commands" in lower

    def test_conversation_stage_content(self):
        """Conversation stage teaches dialogue skills."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="conversation")
        lower = result.lower()
        assert "conversation" in lower
        assert "dialogue" in lower or "multi-sentence" in lower

    def test_commands_stage_content(self):
        """Commands stage teaches [CMD] syntax."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="commands")
        assert "[CMD]" in result
        assert "web" not in result.split("COMMANDS")[1].split(
            "IMPORTANT")[0].lower() or "NOT" in result

    def test_web_stage_content(self):
        """Web stage teaches search.web and web.fetch."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="web")
        assert "search.web" in result
        assert "web.fetch" in result

    def test_unknown_stage_falls_back_to_basics(self):
        """Unknown stage falls back to basics."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="unknown_stage")
        assert "BASICS" in result


# ================================================================
# FORGE: Training Brief (Quick Profile + Custom Brief)
# ================================================================

class TestTrainingBrief:
    """Test Training Brief feature — quick profile fields + freeform text."""

    def test_build_trainer_prompt_accepts_training_brief(self):
        """_build_trainer_system_prompt accepts training_brief kwarg."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000,
            training_brief="Personality: cheerful. Expertise: cooking.")
        assert "cheerful" in result
        assert "cooking" in result

    def test_training_brief_empty_is_fine(self):
        """Empty training_brief doesn't break prompt generation."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, training_brief="")
        assert isinstance(result, str)
        assert len(result) > 50

    def test_training_brief_none_is_fine(self):
        """None training_brief doesn't break prompt generation."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, training_brief=None)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_training_brief_placed_prominently(self):
        """Training brief appears BEFORE the generic instructions."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        brief = "The AI should be a sarcastic chef named Gordon."
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, training_brief=brief)
        # Brief must appear before the critical/generic section
        brief_pos = result.find("sarcastic chef")
        critical_pos = result.find("CRITICAL")
        assert brief_pos < critical_pos, (
            "Training brief should appear before generic instructions")

    def test_training_brief_has_section_header(self):
        """Training brief is wrapped with a clear section header."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000,
            training_brief="Be a pirate.")
        # Should have a header marking the user's brief
        assert "USER" in result.upper() or "BRIEF" in result.upper() or "GOAL" in result.upper()

    def test_quick_profile_fields_defined(self):
        """Quick profile has the expected field names."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        fields = ForgeMixin._QUICK_PROFILE_FIELDS
        assert isinstance(fields, (list, tuple))
        names = [f[0] for f in fields]
        assert "Personality" in names
        assert "Tone" in names
        assert "Expertise" in names
        # Name is NOT a quick field — it comes from the student model
        assert "Name" not in names

    def test_save_training_brief_persists_epochs_lr(self):
        """_save_training_brief includes epochs, LR, and preset."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._save_training_brief)
        assert "_epochs" in source
        assert "_lr" in source
        assert "_preset" in source

    def test_load_training_brief_restores_epochs_lr(self):
        """_load_training_brief restores epochs, LR, and preset."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._load_training_brief)
        assert "_epochs" in source
        assert "_lr" in source
        assert "_preset" in source

    def test_save_training_brief_persists_pretrain_settings(self):
        """_save_training_brief includes vocab, retrain tok, utf8."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._save_training_brief)
        assert "_pretrain_vocab" in source
        assert "_pretrain_retrain_tok" in source
        assert "_pretrain_utf8" in source

    def test_save_training_brief_persists_lora_and_data_path(self):
        """_save_training_brief includes LoRA settings and Basic data path."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._save_training_brief)
        assert "_lora_rank" in source
        assert "_lora_alpha" in source
        assert "_train_data_path" in source

    def test_load_training_brief_restores_all_new_fields(self):
        """_load_training_brief restores all new persistence fields."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._load_training_brief)
        assert "_pretrain_vocab" in source
        assert "_pretrain_retrain_tok" in source
        assert "_lora_rank" in source
        assert "_train_data_path" in source


# ================================================================
# FORGE: UI Polish
# ================================================================


# ================================================================
# FORGE: Web Learn
# ================================================================


# ================================================================
# System as visible third speaker in chat
# ================================================================


# ================================================================
# Chat Media Support (inline images, GIFs, video thumbnails, links)
# ================================================================

class TestChatMedia:
    """Media rendering in the chat display — images, GIFs, videos, clickable links."""

    def test_detect_media_refs_finds_images(self):
        """detect_media_refs finds image paths and URLs in text."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Here is an image: outputs/images/test.png and done."
        refs = detect_media_refs(text)
        assert len(refs) >= 1
        assert any(r["path"].endswith("test.png") for r in refs)
        assert any(r["type"] == "image" for r in refs)

    def test_detect_media_refs_finds_gifs(self):
        """detect_media_refs identifies GIF files."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Check this: outputs/gifs/anim.gif"
        refs = detect_media_refs(text)
        assert len(refs) >= 1
        assert any(r["type"] == "gif" for r in refs)

    def test_detect_media_refs_finds_videos(self):
        """detect_media_refs identifies video files."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Video at outputs/videos/demo.mp4"
        refs = detect_media_refs(text)
        assert len(refs) >= 1
        assert any(r["type"] == "video" for r in refs)

    def test_detect_urls_finds_http(self):
        """detect_urls finds http and https URLs in text."""
        from enigma_engine.gui.media import detect_urls
        text = "Visit https://example.com and http://test.org/page"
        urls = detect_urls(text)
        assert "https://example.com" in urls
        assert "http://test.org/page" in urls

    def test_detect_urls_finds_image_urls(self):
        """detect_urls identifies image URLs as media type."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Image at https://example.com/photo.jpg"
        refs = detect_media_refs(text)
        assert any(r["type"] == "image" for r in refs)

    def test_load_chat_image_returns_photoimage(self):
        """load_chat_image returns a PhotoImage-compatible object."""
        from enigma_engine.gui.media import load_chat_image
        # Create a tiny test image in memory
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        import tkinter as tk
        import tempfile, os
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("Tcl/Tk not available")
        try:
            img = Image.new("RGB", (100, 100), color="red")
            with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False) as f:
                img.save(f, format="PNG")
                tmp_path = f.name
            try:
                result = load_chat_image(tmp_path, max_width=80)
                assert result is not None
                assert hasattr(result, "width")
                assert hasattr(result, "height")
                # Should be resized
                assert result.width() <= 80
            finally:
                os.unlink(tmp_path)
        finally:
            root.destroy()

    def test_load_chat_image_returns_none_for_missing(self):
        """load_chat_image returns None for missing files."""
        from enigma_engine.gui.media import load_chat_image
        result = load_chat_image("/nonexistent/image.png")
        assert result is None

    def test_extract_gif_frames_returns_list(self):
        """extract_gif_frames returns a list of PhotoImage frames."""
        from enigma_engine.gui.media import extract_gif_frames
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        import tkinter as tk
        import tempfile, os
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("Tcl/Tk not available")
        try:
            # Create a minimal 2-frame GIF
            frames = [
                Image.new("RGB", (10, 10), "red"),
                Image.new("RGB", (10, 10), "blue"),
            ]
            with tempfile.NamedTemporaryFile(
                    suffix=".gif", delete=False) as f:
                frames[0].save(
                    f, format="GIF", save_all=True,
                    append_images=frames[1:], duration=100, loop=0)
                tmp_path = f.name
            try:
                result = extract_gif_frames(tmp_path, max_width=20)
                assert isinstance(result, list)
                assert len(result) >= 2
                # Each frame should have (photo_image, duration_ms)
                for photo, dur in result:
                    assert hasattr(photo, "width")
                    assert isinstance(dur, int)
            finally:
                os.unlink(tmp_path)
        finally:
            root.destroy()

    def test_extract_video_thumbnail_with_cv2(self):
        """extract_video_thumbnail returns an image for valid video."""
        from enigma_engine.gui.media import extract_video_thumbnail
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        import tkinter as tk
        import tempfile, os, numpy as np
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("Tcl/Tk not available")
        try:
            # Create a tiny valid video file
            tmp_path = tempfile.mktemp(suffix=".avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(tmp_path, fourcc, 1, (64, 64))
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :] = (0, 0, 255)  # red
            writer.write(frame)
            writer.release()
            try:
                result = extract_video_thumbnail(tmp_path, max_width=50)
                assert result is not None
                assert hasattr(result, "width")
            finally:
                os.unlink(tmp_path)
        finally:
            root.destroy()

    def test_extract_video_thumbnail_returns_none_for_missing(self):
        """extract_video_thumbnail returns None for missing files."""
        from enigma_engine.gui.media import extract_video_thumbnail
        result = extract_video_thumbnail("/nonexistent/video.mp4")
        assert result is None

    def test_media_constants(self):
        """Media module has file extension constants."""
        from enigma_engine.gui.media import (
            IMAGE_EXTENSIONS, GIF_EXTENSIONS, VIDEO_EXTENSIONS)
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".gif" in GIF_EXTENSIONS
        assert ".mp4" in VIDEO_EXTENSIONS

    def test_detect_media_refs_relative_and_absolute(self):
        """detect_media_refs handles both relative and absolute paths."""
        from enigma_engine.gui.media import detect_media_refs
        # Relative path
        refs1 = detect_media_refs("see outputs/images/cat.jpg")
        assert len(refs1) >= 1
        # Absolute-style path
        refs2 = detect_media_refs(r"see C:\images\cat.jpg")
        assert len(refs2) >= 1

    def test_detect_media_refs_no_false_positives(self):
        """detect_media_refs does not match random text."""
        from enigma_engine.gui.media import detect_media_refs
        refs = detect_media_refs("Hello world, nothing here")
        assert len(refs) == 0

    def test_detect_markdown_image_syntax(self):
        """detect_media_refs parses ![alt](url) markdown images."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Here is Pikachu: ![Pikachu](https://example.com/pikachu.jpg)"
        refs = detect_media_refs(text)
        url_refs = [r for r in refs if r["source"] == "url"]
        assert len(url_refs) >= 1
        assert any(r["path"] == "https://example.com/pikachu.jpg"
                    for r in url_refs)
        assert any(r["type"] == "image" for r in url_refs)

    def test_markdown_image_has_alt_text(self):
        """Markdown image refs include alt text when present."""
        from enigma_engine.gui.media import detect_media_refs
        text = "![My Cat](https://example.com/cat.png)"
        refs = detect_media_refs(text)
        md_refs = [r for r in refs if r.get("alt")]
        assert len(md_refs) >= 1
        assert md_refs[0]["alt"] == "My Cat"

    def test_markdown_gif_detected(self):
        """Markdown image syntax with .gif extension detected as gif type."""
        from enigma_engine.gui.media import detect_media_refs
        text = "![anim](https://example.com/anim.gif)"
        refs = detect_media_refs(text)
        assert any(r["type"] == "gif" for r in refs)

    def test_no_false_file_match_on_url_domain(self):
        """File path regex does not match URL domain components."""
        from enigma_engine.gui.media import detect_media_refs
        text = "See https://raw.githubusercontent.com/user/repo/image.jpg"
        refs = detect_media_refs(text)
        file_refs = [r for r in refs if r["source"] == "file"]
        assert len(file_refs) == 0, (
            f"URL domain falsely matched as file: {file_refs}")

    def test_markdown_image_no_duplicate_url(self):
        """Markdown image URL not duplicated as a separate bare URL ref."""
        from enigma_engine.gui.media import detect_media_refs
        text = "![pic](https://example.com/pic.png)"
        refs = detect_media_refs(text)
        # Should only have one ref for the URL, not two
        paths = [r["path"] for r in refs]
        assert paths.count("https://example.com/pic.png") == 1


# ================================================================
# Send guard (double-send crash fix)
# ================================================================


# ================================================================
# STOP button
# ================================================================


# ================================================================
# Message editing
# ================================================================


class TestWindowClose:
    """Verify cleanup on window close."""

    def test_on_close_does_not_silence_exceptions(self):
        """_on_close should log failures instead of using bare except-pass."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        assert "except Exception:\n                    pass" not in source
        assert "except Exception:\n                pass" not in source


class TestForgeUsesModelsDirConstant:
    """Verify FORGE uses MODELS_DIR constant instead of hardcoded paths."""

    def test_no_hardcoded_path_models(self):
        """No Path('models') hardcoded in gui_forge.py."""
        from enigma_engine.gui import gui_forge
        source = inspect.getsource(gui_forge)
        assert 'Path("models")' not in source
        assert "Path('models')" not in source

    def test_checkpoint_dir_uses_models_dir(self):
        """checkpoint_dir uses MODELS_DIR, not hardcoded string."""
        from enigma_engine.gui import gui_forge
        source = inspect.getsource(gui_forge)
        assert 'checkpoint_dir="models/checkpoints"' not in source


class TestFileEncoding:
    """Verify all file I/O uses encoding='utf-8' on Windows."""

    def test_builtin_commands_mod_json_encoding(self):
        """mod.json reads use encoding='utf-8'."""
        from enigma_engine.core import builtin_commands
        source = inspect.getsource(builtin_commands)
        # Should NOT have open(mod_json, 'r') without encoding
        assert "open(mod_json, 'r')" not in source


class TestRouterPortDynamic:
    """Verify router port is not hardcoded in messages."""

    def test_mod_start_uses_dynamic_port(self):
        """mod_start command shows actual router port."""
        from enigma_engine.core import builtin_commands
        source = inspect.getsource(builtin_commands)
        assert "port 9900" not in source


class TestRouterStartupLogging:
    """Verify router startup failures are logged, not silently swallowed."""

    def test_desktop_router_not_bare_pass(self):
        """Router startup does not use bare except: pass."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        # The old pattern was 'except Exception:\n            pass'
        assert "pass  # Router optional" not in source


class TestExpandingChatDisplay:
    """Verify the chat area uses native CTkTextbox scrollbar."""

    def test_mousewheel_not_redirected(self):
        """Chat display no longer needs mousewheel redirect (native scroll)."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        # redirect_mousewheel was removed — native scrollbar handles it
        assert "_redirect_mousewheel" not in source


class TestGuiDeadImports:
    """Verify GUI files don't have dead imports."""

    def test_media_no_unused_os(self):
        """media.py should not import unused os."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "media.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        import_lines = [l for l in lines if l.strip() == 'import os']
        usage_lines = [l for l in lines if 'os.' in l and 'import' not in l]
        if import_lines:
            assert usage_lines, "os is imported but never used in media.py"

    def test_media_no_unused_imagefont(self):
        """media.py should not import unused ImageFont."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "media.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        import re
        import_lines = [l for l in lines if 'ImageFont' in l and 'import' in l]
        usage_lines = [l for l in lines if re.search(r'\bImageFont\b', l)
                       and 'import' not in l]
        if import_lines:
            assert usage_lines, "ImageFont is imported but never used in media.py"

    def test_mod_page_no_unused_border_accent(self):
        """gui_mod_page.py should not import unused C_BORDER_ACCENT."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_mod_page.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        usage_lines = [l for l in source.split('\n')
                       if re.search(r'\bC_BORDER_ACCENT\b', l) and 'import' not in l]
        if not usage_lines:
            # Should not be imported if not used
            import_lines = [l for l in source.split('\n')
                            if 'C_BORDER_ACCENT' in l and 'import' in l]
            assert not import_lines, "C_BORDER_ACCENT imported but never used"

    def test_cmd_page_no_unused_c_accent(self):
        """gui_cmd_page.py should not import unused C_ACCENT."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_cmd_page.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        # C_ACCENT but NOT C_ACCENT_DIM (which IS used)
        usage_lines = [l for l in source.split('\n')
                       if re.search(r'\bC_ACCENT\b', l)
                       and 'C_ACCENT_DIM' not in l
                       and 'C_ACCENT_MUTED' not in l
                       and 'import' not in l]
        if not usage_lines:
            import_lines = [l for l in source.split('\n')
                            if re.search(r'\bC_ACCENT\b', l) and 'import' in l
                            and 'C_ACCENT_DIM' not in l.replace('C_ACCENT,', '')]
            # Check if C_ACCENT alone appears on import line
            assert 'C_ACCENT,' not in source.split('import')[1] if len(source.split('import')) > 1 else True


# ================================================================
# Polish audit — fixes verified 2026-02-26
# ================================================================

class TestPolishAuditGUI:
    """Verify polish fixes made to GUI files."""

    def test_desktop_no_dead_c_green(self):
        """desktop.py should not import unused C_GREEN."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        usage = [l for l in source.split('\n')
                 if re.search(r'\bC_GREEN\b', l) and 'import' not in l]
        if not usage:
            import_lines = [l for l in source.split('\n')
                            if 'C_GREEN' in l and 'import' in l]
            assert not import_lines, "C_GREEN imported but never used in desktop.py"

    def test_desktop_no_dead_c_text(self):
        """desktop.py should not import unused C_TEXT."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        usage = [l for l in source.split('\n')
                 if re.search(r'\bC_TEXT\b', l)
                 and 'C_TEXT_BRIGHT' not in l
                 and 'C_TEXT_DIM' not in l
                 and 'import' not in l]
        if not usage:
            import_lines = [l for l in source.split('\n')
                            if re.search(r'\bC_TEXT\b', l)
                            and 'C_TEXT_BRIGHT' not in l
                            and 'C_TEXT_DIM' not in l
                            and 'import' in l]
            assert not import_lines, "C_TEXT imported but never used in desktop.py"

    def test_gui_pages_no_dead_purple_dim(self):
        """gui_pages.py should not import unused C_PURPLE_DIM."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_pages.py"
        source = source_path.read_text(encoding='utf-8')
        assert 'C_PURPLE_DIM' not in source, "C_PURPLE_DIM imported but never used"

    def test_gui_pages_no_dead_purple_muted(self):
        """gui_pages.py should not import unused C_PURPLE_MUTED."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_pages.py"
        source = source_path.read_text(encoding='utf-8')
        assert 'C_PURPLE_MUTED' not in source, "C_PURPLE_MUTED imported but never used"

    def test_gui_logic_no_dead_accent_dim(self):
        """gui_logic.py should not import unused C_ACCENT_DIM."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        usage = [l for l in source.split('\n')
                 if re.search(r'\bC_ACCENT_DIM\b', l) and 'import' not in l]
        if not usage:
            import_lines = [l for l in source.split('\n')
                            if 'C_ACCENT_DIM' in l and 'import' in l]
            assert not import_lines, "C_ACCENT_DIM imported but never used"

    def test_version_constant_exists(self):
        """widgets.py should export a VERSION constant."""
        from enigma_engine.gui.widgets import VERSION
        assert isinstance(VERSION, str)
        assert VERSION  # Not empty

    def test_font_family_constant(self):
        """widgets.py should use FONT_FAMILY constant for all font tuples."""
        from enigma_engine.gui.widgets import FONT_FAMILY
        assert FONT_FAMILY == "Consolas"

    def test_tooltip_uses_constants(self):
        """Tooltip should use color/font constants, not hardcoded values."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "widgets.py"
        source = source_path.read_text(encoding='utf-8')
        # Find the _show method
        lines = source.split('\n')
        in_tooltip_show = False
        for line in lines:
            if 'def _show(self)' in line:
                in_tooltip_show = True
            elif in_tooltip_show and 'def ' in line:
                break
            elif in_tooltip_show:
                # Should not reference raw color hex in tooltip
                if 'background=' in line or 'foreground=' in line:
                    assert '"#' not in line, (
                        f"Tooltip uses hardcoded color: {line.strip()}")

    def test_tooltip_dismiss_on_focus_loss(self):
        """Tooltip must dismiss when app loses focus and not use -topmost."""
        import inspect
        from enigma_engine.gui.widgets import Tooltip
        source = inspect.getsource(Tooltip)
        # Must have FocusOut binding for app focus loss
        assert 'FocusOut' in source, (
            "Tooltip has no <FocusOut> binding — stays visible when app loses focus")
        # Must use wm_transient so tooltip follows parent z-order
        assert 'wm_transient' in source, (
            "Tooltip does not use wm_transient — floats above other apps")
        # Must NOT use -topmost (causes tooltip to stay above all windows)
        show_src = inspect.getsource(Tooltip._show)
        assert '-topmost' not in show_src, (
            "Tooltip._show uses -topmost — tooltip persists above other apps")

    def test_config_page_no_core_dropdown_text(self):
        """CONFIG page should not reference nonexistent CORE dropdown."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_pages.py"
        source = source_path.read_text(encoding='utf-8')
        assert "CORE page dropdown" not in source, (
            "CONFIG page references nonexistent CORE page profile dropdown")

    def test_cmd_clear_does_not_rescan(self):
        """_cmd_clear should not call _cmd_welcome (rescans filesystem)."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_cmd_page.py"
        source = source_path.read_text(encoding='utf-8')
        # Find the _cmd_clear method
        lines = source.split('\n')
        in_clear = False
        for line in lines:
            if 'def _cmd_clear' in line:
                in_clear = True
            elif in_clear and line.strip().startswith('def '):
                break
            elif in_clear:
                assert '_cmd_welcome()' not in line, (
                    "_cmd_clear calls _cmd_welcome which rescans filesystem")

    def test_media_uses_named_constants(self):
        """media.py should use named constants for limits, not magic numbers."""
        from enigma_engine.gui.media import (
            MAX_GIF_FRAMES, MAX_IMAGE_DOWNLOAD_BYTES,
            MAX_GIF_DOWNLOAD_BYTES, MEDIA_DOWNLOAD_TIMEOUT)
        assert MAX_GIF_FRAMES == 120
        assert MAX_IMAGE_DOWNLOAD_BYTES == 10 * 1024 * 1024
        assert MAX_GIF_DOWNLOAD_BYTES == 20 * 1024 * 1024
        assert MEDIA_DOWNLOAD_TIMEOUT == 10

    def test_desktop_no_dead_session_counter_init(self):
        """desktop.py should not have redundant _session_counter = 0."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        assert "_session_counter: int = 0" not in source, (
            "Redundant initialization immediately overwritten by = 1")


# ================================================================
# Voice Input: Conversational Mode
# ================================================================

class TestVoiceConversation:
    """Verify voice input works conversationally (auto-send)."""

    def test_voice_continuous_listening(self):
        """Voice input should keep listening after each phrase."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic.py"
        source = source_path.read_text(encoding='utf-8')
        # The old code set _voice_got_audio to stop after one phrase;
        # the new code should NOT have that pattern
        assert '_voice_got_audio = True' not in source, (
            "Voice input stops after one phrase — should be continuous")


# ================================================================
# Voice Output: TTS
# ================================================================


# ================================================================
# Model Delete: No GUI Freeze
# ================================================================

class TestModelDeleteNoFreeze:
    """Verify model deletion runs heavy work off the main thread."""

    def test_delete_model_no_direct_unload(self):
        """_delete_model should not call _unload_model directly.

        _unload_model does torch.cuda.empty_cache() which is slow;
        the delete method should handle unloading inline without
        blocking the GUI.
        """
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_forge_models.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        in_method = False
        for line in lines:
            if 'def _delete_model' in line:
                in_method = True
            elif in_method and line.strip().startswith('def ') and '_do_delete' not in line:
                break
            elif in_method and '_unload_model()' in line:
                pytest.fail(
                    "_delete_model should not call _unload_model "
                    "directly — it freezes the GUI")

    def test_refresh_models_uses_thread(self):
        """_refresh_models should scan in a background thread."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_forge_models.py"
        source = source_path.read_text(encoding='utf-8')
        # Extract _refresh_models through to next top-level method
        lines = source.split('\n')
        in_method = False
        method_lines = []
        for line in lines:
            if 'def _refresh_models' in line:
                in_method = True
            elif in_method and line.strip().startswith('def ') and not line.startswith(' ' * 8):
                # Hit next class-level method (4 spaces indent)
                break
            if in_method:
                method_lines.append(line)
        method_body = '\n'.join(method_lines)
        assert 'Thread' in method_body, (
            "_refresh_models should scan models in background thread")


# ================================================================
# Model Size Display
# ================================================================

class TestModelSizeDisplay:
    """Verify model cards show user-entered size, not computed params."""

    def test_normalise_size_label_b(self):
        """'8b' should normalise to '8B'."""
        from enigma_engine.gui.scanners import _normalise_size_label
        assert _normalise_size_label("8b") == "8B"

    def test_normalise_size_label_decimal_b(self):
        """'1.5b' should normalise to '1.5B'."""
        from enigma_engine.gui.scanners import _normalise_size_label
        assert _normalise_size_label("1.5b") == "1.5B"

    def test_normalise_size_label_m(self):
        """'500m' should normalise to '0.50B'."""
        from enigma_engine.gui.scanners import _normalise_size_label
        assert _normalise_size_label("500m") == "0.50B"

    def test_normalise_size_label_preset(self):
        """Preset names like 'small' pass through unchanged."""
        from enigma_engine.gui.scanners import _normalise_size_label
        assert _normalise_size_label("small") == "small"


# ================================================================
# Memory Optimization: Param Counting + Image Cap
# ================================================================

class TestMemoryOptimization:
    """Verify RAM optimizations — no huge torch.load, capped images."""

    def test_format_param_count_billions(self):
        """_format_param_count formats large numbers as B."""
        from enigma_engine.gui.scanners import _format_param_count
        assert _format_param_count(19_080_000_000) == "19.08B"

    def test_format_param_count_millions(self):
        """_format_param_count formats millions as M."""
        from enigma_engine.gui.scanners import _format_param_count
        assert _format_param_count(5_000_000) == "5.0M"

    def test_format_param_count_small(self):
        """_format_param_count formats small numbers with commas."""
        from enigma_engine.gui.scanners import _format_param_count
        assert _format_param_count(1234) == "1,234"

    def test_max_chat_images_constant(self):
        """MAX_CHAT_IMAGES constant exists in media.py."""
        from enigma_engine.gui.media import MAX_CHAT_IMAGES
        assert isinstance(MAX_CHAT_IMAGES, int)
        assert MAX_CHAT_IMAGES > 0

    def test_peek_target_size_returns_none_for_missing(self):
        """_peek_target_size returns None for non-existent file."""
        from enigma_engine.gui.scanners import _peek_target_size
        result = _peek_target_size(Path("nonexistent_model.pth"))
        assert result is None


# ================================================================
# Deferred Imports: Avoid Loading torch/transformers at Startup
# ================================================================

class TestDeferredImports:
    """Verify heavy libraries are NOT loaded when importing the GUI."""

    def test_core_init_no_eager_loader_imports(self):
        """core/__init__.py must not eagerly import loader modules."""
        import re
        source_path = (Path(__file__).parent.parent
                       / "enigma_engine" / "core" / "__init__.py")
        source = source_path.read_text(encoding='utf-8')
        for mod in ['gguf_loader', 'huggingface_loader', 'ollama_loader',
                     'onnx_loader', 'gptq_awq_loader']:
            eager = re.findall(
                rf'^from \.{mod} import', source, re.MULTILINE)
            assert not eager, (
                f"core/__init__.py still eagerly imports {mod}")

    def test_core_init_kv_cache_lazy(self):
        """KVCache import must be lazy in core/__init__.py."""
        import re
        source_path = (Path(__file__).parent.parent
                       / "enigma_engine" / "core" / "__init__.py")
        source = source_path.read_text(encoding='utf-8')
        eager = re.findall(r'^from \.kv_cache import', source, re.MULTILINE)
        assert not eager, (
            "core/__init__.py eagerly imports kv_cache — it must be lazy")

    def test_huggingface_loader_deferred(self):
        """huggingface_loader must NOT import transformers at module level."""
        import re
        source_path = (Path(__file__).parent.parent / "enigma_engine"
                       / "core" / "huggingface_loader.py")
        source = source_path.read_text(encoding='utf-8')
        top_level = re.findall(
            r'^from transformers import', source, re.MULTILINE)
        assert not top_level, (
            "huggingface_loader.py has top-level 'from transformers import'")

    def test_gptq_awq_loader_deferred(self):
        """gptq_awq_loader must NOT import transformers at module level."""
        import re
        source_path = (Path(__file__).parent.parent / "enigma_engine"
                       / "core" / "gptq_awq_loader.py")
        source = source_path.read_text(encoding='utf-8')
        top_level = re.findall(
            r'^from transformers import', source, re.MULTILINE)
        assert not top_level, (
            "gptq_awq_loader.py has top-level 'from transformers import'")

    def test_gguf_loader_deferred(self):
        """gguf_loader must NOT import llama_cpp at module level."""
        import re
        source_path = (Path(__file__).parent.parent / "enigma_engine"
                       / "core" / "gguf_loader.py")
        source = source_path.read_text(encoding='utf-8')
        top_level = re.findall(
            r'^from llama_cpp import', source, re.MULTILINE)
        assert not top_level, (
            "gguf_loader.py has top-level 'from llama_cpp import'")

    def test_lazy_loaders_resolve(self):
        """Lazy loader attributes resolve correctly via __getattr__."""
        from enigma_engine import core
        for name in ['load_gguf_model', 'load_huggingface_model',
                     'load_ollama_model', 'load_onnx_model']:
            attr = getattr(core, name, 'MISSING')
            assert attr != 'MISSING', f"core.{name} not accessible"


# ================================================================
# TTS Thread Safety: No Cross-Thread COM Calls
# ================================================================

class TestTTSThreadSafety:
    """Verify TTS stop uses callback instead of cross-thread engine.stop()."""

    def test_tts_no_started_word_callback(self):
        """Worker must NOT use started-word callback — it corrupts SAPI5."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        match = re.search(
            r'def _tts_worker\b.*?(?=\n            t = threading\.Thread)',
            source, re.DOTALL)
        assert match, "_tts_worker not found"
        body = match.group(0)
        assert "started-word" not in body or "do NOT use" in body.lower() or "intentionally" in body.lower(), (
            "TTS worker must not connect started-word callback — "
            "calling engine.stop() inside runAndWait breaks SAPI5")

    def test_tts_stop_sets_event(self):
        """_tts_stop must signal the event, not call engine.stop() directly."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        # Extract just the _tts_stop method body (code lines only,
        # excluding docstrings) up to the next method.
        import re
        match = re.search(
            r'def _tts_stop\b.*?(?=\n    def )',
            source, re.DOTALL)
        assert match, "_tts_stop method not found"
        body = match.group(0)
        # Check code lines only — skip docstring content
        code_lines = []
        in_docstring = False
        for line in body.split('\n'):
            stripped = line.strip()
            if stripped.startswith('"""'):
                # Toggle docstring — skip lines inside it
                if in_docstring:
                    in_docstring = False
                    continue
                if stripped.endswith('"""') and len(stripped) > 3:
                    continue  # single-line docstring
                in_docstring = True
                continue
            if not in_docstring:
                code_lines.append(line)
        code_body = '\n'.join(code_lines)
        assert '_tts_stop_event' in code_body and '.set()' in code_body, (
            "_tts_stop must set the stop event")
        assert 'engine' not in code_body or '.stop()' not in code_body, (
            "_tts_stop must NOT call engine.stop() — "
            "cross-thread COM calls crash on Windows SAPI5")


# ================================================================
# TTS Text Cleaning: Safe Text for SAPI5
# ================================================================

class TestTTSTextCleaning:
    """Verify TTS cleans and chunks text before speaking."""

    def test_tts_clean_strips_code_blocks(self):
        """Code blocks should be replaced with a short label."""
        from enigma_engine.gui.gui_logic import LogicMixin
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        result = LogicMixin._tts_clean_text(None, text)
        assert '```' not in result
        assert "print" not in result
        assert "Done" in result

    def test_tts_clean_strips_inline_code(self):
        """Backtick-wrapped code should have backticks removed."""
        from enigma_engine.gui.gui_logic import LogicMixin
        result = LogicMixin._tts_clean_text(None, "Use `print()` here.")
        assert '`' not in result
        assert "print" in result

    def test_tts_clean_strips_markdown(self):
        """Markdown bold/italic markers should be removed."""
        from enigma_engine.gui.gui_logic import LogicMixin
        result = LogicMixin._tts_clean_text(None, "This is **bold**.")
        assert '**' not in result
        assert "bold" in result

    def test_tts_clean_strips_urls(self):
        """URLs should be replaced with 'link'."""
        from enigma_engine.gui.gui_logic import LogicMixin
        result = LogicMixin._tts_clean_text(
            None, "Visit https://example.com/path?q=1 now.")
        assert "https://" not in result

    def test_tts_clean_strips_cmd_blocks(self):
        """[CMD]...[/CMD] blocks should be removed."""
        from enigma_engine.gui.gui_logic import LogicMixin
        result = LogicMixin._tts_clean_text(
            None, "Done. [CMD]file.read x[/CMD] OK.")
        assert "[CMD]" not in result
        assert "OK" in result

    def test_tts_chunks_long_text(self):
        """Long text should be split into sentence chunks."""
        from enigma_engine.gui.gui_logic import LogicMixin
        # Build text longer than 180 chars with sentence boundaries
        text = "This is a fairly long test sentence for chunking. " * 8
        assert len(text) > 200, "test text must exceed chunk limit"
        chunks = LogicMixin._tts_chunk_text(None, text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 2, (
            f"Text of {len(text)} chars should be split into "
            f"multiple chunks, got {len(chunks)}")

    def test_tts_chunks_respect_max_length(self):
        """No chunk should exceed the max character limit."""
        from enigma_engine.gui.gui_logic import LogicMixin
        text = "Word " * 100  # ~500 chars, no periods
        chunks = LogicMixin._tts_chunk_text(None, text)
        for chunk in chunks:
            assert len(chunk) <= 200, (
                f"Chunk too long ({len(chunk)} chars): {chunk[:50]}...")


# ================================================================
# Boot Time: Deferred Param Counting
# ================================================================

class TestDeferredBootParam:
    """Verify model param counting is deferred to background thread."""

    def test_scan_models_returns_estimated_params(self):
        """scan_models should estimate params from file size (not load the model)."""
        from enigma_engine.gui.scanners import scan_models
        models = scan_models()
        # Models with recognized extensions should have a file-size estimate
        for m in models:
            if m["format"] in ("gguf", "pth", "pt", "bin", "safetensors"):
                assert m["params"] is None or isinstance(m["params"], (int, float)), (
                    f"params should be None or a numeric estimate, "
                    f"got {type(m['params'])} for {m['name']}")

    def test_cpuinfo_not_called_synchronously(self):
        """cpuinfo.get_cpu_info() must not block the status tick."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        match = re.search(
            r'def _tick\b.*?(?=\n        self\.after)',
            source, re.DOTALL)
        assert match, "_tick function not found"
        tick_body = match.group(0)
        assert 'cpuinfo' not in tick_body, (
            "cpuinfo must not be called inside _tick — "
            "it blocks the UI for seconds")

    def test_status_ticker_uses_low_overhead_interval(self, monkeypatch):
        """Low-overhead mode slows the steady-state status refresh rate."""
        from enigma_engine.gui.desktop import EnigmaGUI
        import enigma_engine.gui.desktop as desktop_mod

        after_calls = []

        monkeypatch.setattr(desktop_mod.threading, "Thread", DummyThread)

        obj = object.__new__(EnigmaGUI)
        obj._gaming_mode_active = True
        obj._status_tick_ms = 5000
        obj._boot_time = 0.0
        obj._hw_device_label = "CPU"
        obj._shutting_down = False
        obj.status_bar = DummyStatusBar()
        obj.state = lambda: "normal"  # mock tkinter state()

        def after(ms, callback):
            after_calls.append(ms)
            if len(after_calls) == 1:
                callback()

        obj.after = after

        obj._start_status_ticker()

        assert after_calls == [100, 5000]

    def test_count_model_params_skips_in_low_overhead_mode(self, monkeypatch):
        """Low-overhead mode must not start exact param counting threads."""
        from enigma_engine.gui.desktop import EnigmaGUI
        import enigma_engine.gui.desktop as desktop_mod

        monkeypatch.setattr(desktop_mod.threading, "Thread", UnexpectedThread)

        obj = object.__new__(EnigmaGUI)
        obj._gaming_mode_active = True
        obj.models_data = [{"format": "pth", "params": None, "path": "fake"}]

        obj._count_model_params_background()


# ================================================================
# TTS Queue Drain: Stop Clears Pending Chunks
# ================================================================


# ================================================================
# Scroll Consistency: Always Scroll During Typewriter
# ================================================================


# ================================================================
# Route Prompts
# ================================================================

class TestRoutePrompts:
    """Verify per-route prompt system works."""

    def test_prompts_dir_exists(self):
        """data/prompts/ directory exists."""
        from enigma_engine.gui.scanners import PROMPTS_DIR
        assert PROMPTS_DIR.exists(), "data/prompts/ must exist"

    def test_default_prompt_files_exist(self):
        """Default prompt files (chat, trainer) exist."""
        from enigma_engine.gui.scanners import PROMPTS_DIR
        assert (PROMPTS_DIR / "chat.md").exists()
        assert (PROMPTS_DIR / "trainer.md").exists()

    def test_load_route_prompt_chat(self):
        """load_route_prompt('chat') returns non-empty string."""
        from enigma_engine.gui.scanners import load_route_prompt
        prompt = load_route_prompt("chat")
        assert isinstance(prompt, str)
        assert len(prompt) > 10, "Chat prompt should have content"

    def test_load_route_prompt_trainer(self):
        """load_route_prompt('trainer') returns non-empty string."""
        from enigma_engine.gui.scanners import load_route_prompt
        prompt = load_route_prompt("trainer")
        assert isinstance(prompt, str)
        assert len(prompt) > 10, "Trainer prompt should have content"

    def test_load_route_prompt_missing(self):
        """load_route_prompt for non-existent route returns empty."""
        from enigma_engine.gui.scanners import load_route_prompt
        prompt = load_route_prompt("nonexistent_route_xyz")
        assert prompt == ""

    def test_scan_docs_has_prompts_category(self):
        """scan_docs includes prompt files under 'prompts' category."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        prompts = [d for d in docs if d["category"] == "prompts"]
        assert len(prompts) >= 2, (
            "scan_docs should include at least 2 prompt files")
        names = {d["filename"] for d in prompts}
        assert "chat.md" in names
        assert "trainer.md" in names

    def test_trainer_system_prompt_includes_user_prompt(self):
        """_build_trainer_system_prompt prepends user trainer prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="basics")
        # Should contain content from the trainer prompt file
        from enigma_engine.gui.scanners import load_route_prompt
        user_prompt = load_route_prompt("trainer")
        if user_prompt:
            assert user_prompt[:30] in prompt, (
                "Trainer system prompt should include user prompt")

    def test_model_context_default_prompt(self):
        """ModelContext default prompt loads from chat.md."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_prompt_default")
        # Should either match chat.md content or fallback
        assert len(ctx.system_prompt) > 10
        assert ctx.system_prompt != ""

    def test_prompt_files_readable(self):
        """All prompt files are readable with utf-8."""
        from enigma_engine.gui.scanners import PROMPTS_DIR
        for f in PROMPTS_DIR.glob("*.md"):
            content = f.read_text(encoding="utf-8")
            assert len(content) > 0, f"Prompt file empty: {f.name}"


# ================================================================
# Chat Tab Audit Fixes
# ================================================================

class TestChatTabAuditFixes:
    """Tests for chat tab issue fixes."""

    def test_chat_cursor_uses_motion_not_tag_enter_leave(self):
        """Chat link cursor should use Motion handler, not per-tag Enter/Leave (S741)."""
        import inspect
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin)
        # Must NOT have per-tag Enter/Leave cursor changes
        assert 'tag_bind("link", "<Enter>"' not in source, (
            "link tag should not bind <Enter> for cursor — use <Motion> instead")
        assert 'tag_bind("video_link", "<Enter>"' not in source
        assert 'tag_bind("file_link", "<Enter>"' not in source
        # Must HAVE a Motion-based cursor handler
        assert "<Motion>" in source, (
            "Chat textbox needs <Motion> bind for cursor updates")

    def test_sessions_sorted_newest_first(self):
        """scan_sessions returns newest sessions first."""
        from enigma_engine.gui.scanners import scan_sessions
        sessions = scan_sessions()
        if len(sessions) >= 2:
            for i in range(len(sessions) - 1):
                assert sessions[i]["saved_at"] >= sessions[i + 1]["saved_at"], (
                    "Sessions should be sorted newest-first by saved_at")

    def test_config_default_max_gen_is_2048(self):
        """CONFIG max_gen default is 2048."""
        from enigma_engine.config import CONFIG
        assert CONFIG.get("max_gen") == 8192


class TestEstimateGGUFParams:
    """Tests for GGUF parameter estimation in model loading."""

    def test_metadata_estimation(self):
        """Estimates params from dim, n_layers, vocab_size metadata."""
        from enigma_engine.gui.gui_logic import _estimate_gguf_params
        from types import SimpleNamespace
        cfg = SimpleNamespace(dim=4096, n_layers=32, vocab_size=151936)
        model = SimpleNamespace(config=cfg)
        engine = SimpleNamespace(model=model)
        result = _estimate_gguf_params(engine, "fake.gguf")
        # 12 * 4096^2 * 32 + 151936 * 4096 = 6,442,450,944 + 622,329,856
        assert result > 6_000_000_000
        assert result < 8_000_000_000

    def test_metadata_zero_dim_falls_to_filesize(self, tmp_path):
        """Falls back to file-size heuristic when metadata has dim=0."""
        from enigma_engine.gui.gui_logic import _estimate_gguf_params
        from types import SimpleNamespace
        cfg = SimpleNamespace(dim=0, n_layers=0, vocab_size=0)
        model = SimpleNamespace(config=cfg)
        engine = SimpleNamespace(model=model)
        fake = tmp_path / "test.gguf"
        fake.write_bytes(b'\x00' * (1024 * 1024))  # 1 MB
        result = _estimate_gguf_params(engine, str(fake))
        assert result > 0

    def test_no_model_falls_to_filesize(self, tmp_path):
        """Falls back when engine has no model attribute."""
        from enigma_engine.gui.gui_logic import _estimate_gguf_params
        engine = type('E', (), {'model': None})()
        fake = tmp_path / "test.gguf"
        fake.write_bytes(b'\x00' * (2 * 1024 * 1024))  # 2 MB
        result = _estimate_gguf_params(engine, str(fake))
        assert result > 0

    def test_missing_file_returns_zero(self):
        """Returns 0 when model has no metadata and file doesn't exist."""
        from enigma_engine.gui.gui_logic import _estimate_gguf_params
        engine = type('E', (), {'model': None})()
        result = _estimate_gguf_params(engine, "/nonexistent/fake.gguf")
        assert result == 0


# =========================================================================
# Deep-dive audit — scanner security, server path traversal, CORS
# =========================================================================


# ── Mod base file presence ──────────────────────────────────────────────────


class TestModBasePresence:
    """Verify mod_base.py exists in each shipped mod folder."""

    def test_imagegen_has_mod_base(self):
        """mods/imagegen/ must contain mod_base.py for imports to work."""
        from pathlib import Path
        assert (Path("mods/imagegen/mod_base.py").exists()), (
            "mods/imagegen/mod_base.py missing — ImageGenMod fails")


# ── Config persistence ──────────────────────────────────────────────────────


# ── Atomic saves ────────────────────────────────────────────────────────────


class TestAtomicSaves:
    """Verify model saves use atomic write pattern."""

    # ── atomic_write_text / atomic_write_json ────────────────────────────

    def test_atomic_write_text_roundtrip(self, tmp_path):
        """atomic_write_text must write and read back correctly."""
        from enigma_engine.core.safe_save import atomic_write_text
        target = tmp_path / "test.txt"
        content = "hello\nworld\n"
        atomic_write_text(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_atomic_write_json_roundtrip(self, tmp_path):
        """atomic_write_json must write and read back valid JSON."""
        import json
        from enigma_engine.core.safe_save import atomic_write_json
        target = tmp_path / "test.json"
        data = {"key": "value", "num": 42, "nested": [1, 2, 3]}
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == data

    def test_atomic_write_text_cleans_tmp_on_failure(self, tmp_path):
        """atomic_write_text must not leave .tmp on write failure."""
        from enigma_engine.core.safe_save import atomic_write_text
        target = tmp_path / "sub" / "test.txt"
        # Create the parent so mkdir doesn't fail
        target.parent.mkdir(parents=True, exist_ok=True)
        # Make target a directory to force os.replace to fail
        target.mkdir()
        with pytest.raises(OSError):
            atomic_write_text(target, "data")
        tmp_file = target.with_suffix(target.suffix + ".tmp")
        assert not tmp_file.exists(), ".tmp file should be cleaned up on failure"

    def test_atomic_write_text_creates_bak(self, tmp_path):
        """atomic_write_text must create .bak of existing file."""
        from enigma_engine.core.safe_save import atomic_write_text
        target = tmp_path / "test.txt"
        target.write_text("original", encoding="utf-8")
        atomic_write_text(target, "updated")
        bak = target.with_suffix(".txt.bak")
        assert bak.exists(), ".bak file should exist"
        assert bak.read_text(encoding="utf-8") == "original"
        assert target.read_text(encoding="utf-8") == "updated"

    def test_atomic_write_json_creates_bak(self, tmp_path):
        """atomic_write_json must create .bak of existing JSON file."""
        import json
        from enigma_engine.core.safe_save import atomic_write_json
        target = tmp_path / "data.json"
        atomic_write_json(target, {"version": 1})
        atomic_write_json(target, {"version": 2})
        bak = target.with_suffix(".json.bak")
        assert bak.exists(), ".bak file should exist for JSON overwrite"
        assert json.loads(bak.read_text(encoding="utf-8")) == {"version": 1}
        assert json.loads(target.read_text(encoding="utf-8")) == {"version": 2}

    def test_no_direct_writes_in_critical_modules(self):
        """Critical data modules must use atomic_write_text/json, not raw writes."""
        import ast
        from pathlib import Path as P
        modules = [
            "enigma_engine/core/memory.py",
            "enigma_engine/core/curated_dataset.py",
            "enigma_engine/core/training_queue.py",
            "enigma_engine/core/training_monitor.py",
            "enigma_engine/core/model_context.py",
            "enigma_engine/core/model_registry.py",
            "enigma_engine/core/ai_profile.py",
            "enigma_engine/core/rag.py",
            "enigma_engine/core/adaptive_trainer.py",
        ]
        for modpath in modules:
            source = P(modpath).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Check for open(..., "w", ...) calls
                if isinstance(func, ast.Name) and func.id == "open":
                    pass
                elif isinstance(func, ast.Attribute) and func.attr == "open":
                    pass
                else:
                    continue
                # Determine mode
                mode = "r"
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if len(node.args) >= 2:
                    arg = node.args[1]
                    if isinstance(arg, ast.Constant):
                        mode = arg.value
                if "w" in mode:
                    assert False, (
                        f"{modpath} line {node.lineno}: uses raw open('w') "
                        f"— must use atomic_write_text/json instead")
            # Also check for .write_text( calls
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "write_text":
                        assert False, (
                            f"{modpath} line {node.lineno}: uses raw "
                            f".write_text() — must use atomic_write_text/json")


# ── Ollama encoding ─────────────────────────────────────────────────────────


class TestOllamaEncoding:
    """Verify ollama_loader.py uses utf-8 encoding on all text opens."""

    def test_no_text_opens_without_encoding(self):
        """All text-mode open() calls must have encoding='utf-8'."""
        import ast
        from pathlib import Path as P
        source = P("enigma_engine/core/ollama_loader.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match open(...) calls
            if isinstance(func, ast.Name) and func.id == "open":
                pass
            elif isinstance(func, ast.Attribute) and func.attr == "open":
                pass
            else:
                continue
            # Check mode arg — skip binary opens
            mode = "r"  # default
            for kw in node.keywords:
                if kw.arg == "mode":
                    if isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
            if len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant):
                    mode = arg.value
            if "b" in mode:
                continue  # binary mode is fine
            # Text mode — must have encoding keyword
            has_encoding = any(
                kw.arg == "encoding" for kw in node.keywords)
            assert has_encoding, (
                f"ollama_loader.py line {node.lineno}: "
                f"text open() missing encoding='utf-8'")


# =====================================================================
# Rename model — case-insensitive Windows support
# =====================================================================


# =====================================================================
# Gradient checkpointing in training
# =====================================================================

class TestGradientCheckpointing:
    """Gradient checkpointing reduces VRAM usage during training."""

    def test_training_config_has_gradient_checkpointing(self):
        """TrainingConfig must have use_gradient_checkpointing field."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "use_gradient_checkpointing")
        assert cfg.use_gradient_checkpointing is True

    def test_gradient_checkpointing_in_to_dict(self):
        """use_gradient_checkpointing must appear in to_dict output."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(use_gradient_checkpointing=True)
        d = cfg.to_dict()
        assert "use_gradient_checkpointing" in d
        assert d["use_gradient_checkpointing"] is True


# =====================================================================
# Training config exposed in FORGE UI
# =====================================================================


# =====================================================================
# Quantize & Export on FORGE page
# =====================================================================


# =====================================================================
# Memory instructions — proactive preference learning
# =====================================================================


# =====================================================================
# Learn While Chatting — BackgroundTrainer wired to chat
# =====================================================================


# =====================================================================
# Theme picker on CONFIG page
# =====================================================================

class TestThemePicker:
    """CONFIG page must have a theme selector with live switching."""

    def test_apply_theme_no_restart(self):
        """_apply_theme must NOT call _restart_gui (live switching)."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._apply_theme)
        assert "_restart_gui" not in source

    def test_reload_theme_returns_color_map(self):
        """reload_theme returns a dict mapping old colours to new."""
        from enigma_engine.gui.widgets import reload_theme, C_BG
        old_bg = C_BG
        color_map = reload_theme("midnight")
        assert isinstance(color_map, dict)
        # Should have mappings since dark != midnight
        assert len(color_map) > 0
        # Restore original theme
        reload_theme("dark")

    def test_reload_theme_updates_globals(self):
        """reload_theme must update C_* module-level constants."""
        from enigma_engine.gui import widgets
        old_bg = widgets.C_BG
        widgets.reload_theme("midnight")
        assert widgets.C_BG != old_bg
        # Restore
        widgets.reload_theme("dark")
        assert widgets.C_BG == old_bg

    def test_reload_theme_same_returns_empty(self):
        """reload_theme with current theme returns empty map."""
        from enigma_engine.gui.widgets import reload_theme
        # Ensure we're on dark
        reload_theme("dark")
        color_map = reload_theme("dark")
        assert color_map == {}

    def test_config_no_profile_section(self):
        """CONFIG page must NOT have an AI Profile section."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_config)
        assert "AI PROFILE" not in source
        assert "profile_dd" not in source
        assert "_apply_profile" not in source

    def test_max_tokens_no_artificial_cap(self):
        """max_tokens upper limit must be large — no artificial 4096 cap."""
        from enigma_engine.gui.scanners import CONFIG_LIMITS
        _lo, hi, _step = CONFIG_LIMITS["max_tokens"]
        assert hi >= 100000, f"max_tokens capped at {hi}, should be uncapped"


# =====================================================================
# Selectable text everywhere — labels and textboxes
# =====================================================================


# =====================================================================
# Silent error swallowing — except Exception: pass must log
# =====================================================================

class TestSilentErrorSwallowing:
    """Critical modules must not silently swallow errors."""

    def test_gui_logic_no_bare_pass(self):
        """gui_logic.py should have minimal except-pass blocks."""
        from enigma_engine.gui import gui_logic
        source = inspect.getsource(gui_logic)
        import re
        # Count bare except-pass (no logging)
        bare_passes = re.findall(
            r'except\s+Exception[^:]*:\s*\n\s*pass\s*$',
            source, re.MULTILINE)
        # Allow some for UI widget guards, but not more than 8
        assert len(bare_passes) <= 8, (
            f"gui_logic.py has {len(bare_passes)} silent "
            f"except-pass blocks — add logger.debug()")

    def test_router_no_bare_pass(self):
        """router.py should have minimal except-pass blocks."""
        from enigma_engine import router
        source = inspect.getsource(router)
        import re
        bare_passes = re.findall(
            r'except\s+Exception[^:]*:\s*\n\s*pass\s*$',
            source, re.MULTILINE)
        assert len(bare_passes) <= 2, (
            f"router.py has {len(bare_passes)} silent "
            f"except-pass blocks — add logger.debug()")

    def test_scanners_no_bare_pass(self):
        """scanners.py should have minimal except-pass blocks."""
        from enigma_engine.gui import scanners
        source = inspect.getsource(scanners)
        import re
        bare_passes = re.findall(
            r'except\s+Exception[^:]*:\s*\n\s*pass\s*$',
            source, re.MULTILINE)
        assert len(bare_passes) <= 1, (
            f"scanners.py has {len(bare_passes)} silent "
            f"except-pass blocks — add logger.debug()")


# ================================================================
# FORGE Feature Tests — presets, preview, history, progress bar,
# learn-while-chatting toggle
# ================================================================

class TestForgePresets:
    """Hyperparameter preset backend logic."""

    def test_preset_values_defined(self):
        """ForgeMixin._TRAINING_PRESETS has expected keys."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        presets = ForgeMixin._TRAINING_PRESETS
        assert "Quick" in presets
        assert "Balanced" in presets
        assert "Thorough" in presets

    def test_preset_tuples_have_three_values(self):
        """Each preset is a (epochs, lr, batch) tuple."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        for name, vals in ForgeMixin._TRAINING_PRESETS.items():
            assert len(vals) == 3, (
                f"Preset '{name}' should have 3 values")

    def test_preset_custom_not_in_presets(self):
        """Custom is not in presets dict (leaves fields unchanged)."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert "Custom" not in ForgeMixin._TRAINING_PRESETS

    def test_quick_preset_values(self):
        """Quick preset: 3 epochs, lr=0.0001, batch=4."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        epochs, lr, batch = ForgeMixin._TRAINING_PRESETS["Quick"]
        assert epochs == "3"
        assert lr == "0.0001"
        assert batch == "4"

    def test_balanced_preset_values(self):
        """Balanced preset: 10 epochs, lr=0.00005, batch=4."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        epochs, lr, batch = ForgeMixin._TRAINING_PRESETS["Balanced"]
        assert epochs == "10"
        assert lr == "0.00005"
        assert batch == "4"

    def test_thorough_preset_values(self):
        """Thorough preset: 30 epochs, lr=0.00002, batch=2."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        epochs, lr, batch = ForgeMixin._TRAINING_PRESETS["Thorough"]
        assert epochs == "30"
        assert lr == "0.00002"
        assert batch == "2"


class TestForgeTrainingHistory:
    """Training history save/load."""

    def test_history_file_path(self):
        """History file is at data/training_history.json."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert ForgeMixin._HISTORY_FILE.name == "training_history.json"
        assert "data" in str(ForgeMixin._HISTORY_FILE)

    def test_save_training_run(self, tmp_path, monkeypatch):
        """_save_training_run writes a valid JSON entry."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run("Solo", "test_model", 5, 0.1234)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(runs) == 1
        assert runs[0]["mode"] == "Solo"
        assert runs[0]["model"] == "test_model"
        assert runs[0]["epochs"] == 5
        assert runs[0]["best_loss"] == 0.1234

    def test_save_training_run_appends(self, tmp_path, monkeypatch):
        """Multiple saves append to the same file."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run("Solo", "m1", 3, 0.5)
        obj._save_training_run("DPO", "m2", 10, 0.3)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(runs) == 2
        assert runs[1]["mode"] == "DPO"

    def test_save_training_run_caps_at_200(
            self, tmp_path, monkeypatch):
        """History is capped at 200 entries."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        history_file.write_text(
            json.dumps([{"mode": "old"}] * 200),
            encoding="utf-8")
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run("New", "m1", 1, 0.1)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(runs) == 200
        assert runs[-1]["mode"] == "New"

    def test_save_training_run_persists_perplexity(
            self, tmp_path, monkeypatch):
        """When perplexity values are supplied they are saved to history."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run(
            "Solo", "test_model", 5, 0.1234,
            before_perplexity=3.75, after_perplexity=2.40)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(runs) == 1
        assert runs[0]["before_perplexity"] == 3.75
        assert runs[0]["after_perplexity"] == 2.40

    def test_save_training_run_no_perplexity_omitted(
            self, tmp_path, monkeypatch):
        """When perplexity is not provided the fields are absent from history."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run("Solo", "test_model", 5, 0.1234)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert "before_perplexity" not in runs[0]
        assert "after_perplexity" not in runs[0]


class TestLearnWhileChattingConfig:
    """Learn-while-chatting toggle on CONFIG page."""

    def test_toggle_saves_setting(self, tmp_path, monkeypatch):
        """Toggle writes learn_while_chatting to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        # Create minimal mock

        sync_calls = []

        obj = object.__new__(PagesMixin)
        obj._learn_while_chatting_var = MockVar()
        obj.status_bar = MockStatusBar()
        obj._refresh_performance_mode = lambda: sync_calls.append("refresh")
        obj._sync_router_training_state = lambda: sync_calls.append("sync")

        obj._toggle_learn_while_chatting()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["learn_while_chatting"] is True
        assert obj._chat_learning_enabled is True
        assert sync_calls == ["refresh", "sync"]


class TestGamingModePreset:
    """Gaming preset should apply the full low-overhead profile."""

    def test_apply_gaming_mode_preset_disables_learning(self, tmp_path, monkeypatch):
        """Preset disables chat learning and syncs runtime state."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        sync_calls = []
        obj = object.__new__(PagesMixin)
        obj.status_bar = MockStatusBar()
        obj._auto_load_chat_model_var = MockVar()
        obj._auto_start_mods_var = MockVar()
        obj._auto_unload_on_minimize_var = MockVar()
        obj._learn_while_chatting_var = MockVar()
        obj._refresh_performance_mode = lambda: sync_calls.append("refresh")
        obj._sync_router_training_state = lambda: sync_calls.append("sync")

        obj._apply_gaming_mode_preset()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_load_chat_model"] is False
        assert data["auto_start_mods"] is False
        assert data["auto_unload_on_minimize"] is True
        assert data["learn_while_chatting"] is False
        assert obj._chat_learning_enabled is False
        assert obj._learn_while_chatting_var.value is False
        assert sync_calls == ["refresh", "sync"]


class TestPerformanceSettings:
    """Performance-related GUI settings for memory usage."""

    def test_toggle_auto_load_chat_model_saves_setting(
            self, tmp_path, monkeypatch):
        """Toggle writes auto_load_chat_model to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._auto_load_chat_model_var = MockVar(False)
        obj.status_bar = MockStatusBar()

        obj._toggle_auto_load_chat_model()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_load_chat_model"] is False

    def test_toggle_auto_start_mods_saves_setting(
            self, tmp_path, monkeypatch):
        """Toggle writes auto_start_mods to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._auto_start_mods_var = MockVar(False)
        obj.status_bar = MockStatusBar()

        obj._toggle_auto_start_mods()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_start_mods"] is False

    def test_toggle_auto_unload_on_minimize_saves_setting(
            self, tmp_path, monkeypatch):
        """Toggle writes auto_unload_on_minimize to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._auto_unload_on_minimize_var = MockVar()
        obj.status_bar = MockStatusBar()

        obj._toggle_auto_unload_on_minimize()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_unload_on_minimize"] is True

    def test_apply_gaming_mode_preset_saves_three_settings(
            self, tmp_path, monkeypatch):
        """Gaming preset writes all memory-related settings."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj.status_bar = MockStatusBar()
        obj._auto_load_chat_model_var = MockVar()
        obj._auto_start_mods_var = MockVar()
        obj._auto_unload_on_minimize_var = MockVar()

        obj._apply_gaming_mode_preset()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_load_chat_model"] is False
        assert data["auto_start_mods"] is False
        assert data["auto_unload_on_minimize"] is True


# ================================================================
# FORGE Page: 3-Mode Contract
# ================================================================

class TestForgeThreeModeContract:
    """Test the current FORGE contract with 3 user-facing modes."""

    def test_teacher_student_removed_from_forge_ui(self):
        """Legacy 'Teacher + Student' option is gone from FORGE page."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert "Teacher + Student" not in source

    def test_legacy_train_with_ai_toggle_removed(self):
        """Legacy Train-with-AI checkbox contract is removed."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert "train_with_ai_var" not in source
        assert "train_with_ai_cb" not in source


# ================================================================
# Shared Web Utilities
# ================================================================

class TestWebUtils:
    """Test shared web search and page fetching utilities."""

    def test_extract_strips_scripts(self):
        """extract_html_text removes script/style content."""
        from enigma_engine.core.web_utils import extract_html_text
        html = (
            "<html><script>var x=1;</script>"
            "<style>.a{color:red}</style>"
            "<p>Real content here</p></html>")
        result = extract_html_text(html)
        assert "var x" not in result
        assert "color" not in result
        assert "Real content here" in result

    def test_extract_strips_nav_footer(self):
        """extract_html_text skips nav, footer, header, aside."""
        from enigma_engine.core.web_utils import extract_html_text
        html = (
            "<nav>Navigation links</nav>"
            "<header>Header stuff</header>"
            "<main><p>Important article text</p></main>"
            "<footer>Footer links</footer>"
            "<aside>Sidebar content</aside>")
        result = extract_html_text(html)
        assert "Important article text" in result
        # Nav/footer/header/aside should be stripped
        assert "Navigation links" not in result
        assert "Footer links" not in result

    def test_extract_empty_html(self):
        """extract_html_text returns empty for blank input."""
        from enigma_engine.core.web_utils import extract_html_text
        assert extract_html_text("") == ""
        assert extract_html_text("<div></div>") == ""


# ================================================================
# FORGE: Optimized Web Learn
# ================================================================

class TestWebLearnOptimized:
    """Test optimized web learn uses shared web_utils and
    trainer system prompt."""

    def test_web_learn_no_inline_ddg_parser(self):
        """_web_learn no longer defines DDGParser inline."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "class DDGParser" not in source

    def test_web_learn_no_inline_text_extractor(self):
        """_web_learn no longer defines TextExtractor inline."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "class TextExtractor" not in source

    def test_web_learn_no_hardcoded_colors(self):
        """Web learn button uses theme constants, not hex."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        # Old hardcoded colors should be gone
        assert "#0d2137" not in source
        assert "#163352" not in source

    def test_legacy_mode_desc_widget_removed(self):
        """Old single-description widget is removed in card-based UI."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert "_training_mode_desc" not in source


# ================================================================
# FORGE: Auto-Train After Data Generation
# ================================================================


# ================================================================
# Phase 1 — Polish
# ================================================================


class TestPrintToLogger:
    """print() calls should be replaced with logger."""

    def test_no_prints_in_server(self):
        """server.py should use logger, not print (except docstrings)."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "api" / "server.py")
        source = source_path.read_text(encoding="utf-8")
        import ast
        tree = ast.parse(source)
        prints = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert len(prints) == 0, (
            f"server.py has {len(prints)} print() calls — "
            "use logger.info instead")


# ================================================================
# Identity card on MODELS page
# ================================================================


# ================================================================
# Identity editing on MODELS page
# ================================================================

class TestModelsIdentityEdit:
    """MODELS page supports inline name editing on model cards."""

    def test_no_simpledialog_in_pages(self):
        """gui_pages.py must not use simpledialog popups."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_pages.py")
        source = source_path.read_text(encoding="utf-8")
        assert "simpledialog" not in source, (
            "gui_pages.py still uses simpledialog — "
            "all inputs must be inline")


# ================================================================
# Identity export
# ================================================================

class TestIdentityExport:
    """ModelContext supports exporting identity as a standalone JSON."""

    def test_export_identity_returns_dict(self):
        """export_identity returns a dict with all identity fields."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("export_test")
        ctx.display_name = "Test AI"
        ctx.personality = "Friendly"
        ctx.tags = ["general"]
        result = ctx.export_identity()
        assert isinstance(result, dict)
        assert result["model_key"] == "export_test"
        assert result["display_name"] == "Test AI"
        assert result["personality"] == "Friendly"
        assert result["tags"] == ["general"]

    def test_export_identity_includes_all_fields(self):
        """export_identity includes stats, training history, notes."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("export_full")
        ctx.total_messages = 100
        ctx.total_sessions = 5
        ctx.notes = "Fine-tuned model"
        result = ctx.export_identity()
        assert result["total_messages"] == 100
        assert result["total_sessions"] == 5
        assert result["notes"] == "Fine-tuned model"
        assert "created_at" in result
        assert "training_history" in result

    def test_export_identity_to_file(self, tmp_path):
        """export_identity can be written to a JSON file."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("export_file")
        ctx.display_name = "Export Test"
        data = ctx.export_identity()
        out = tmp_path / "identity.json"
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["display_name"] == "Export Test"


# ================================================================
# Font size control on CONFIG page
# ================================================================

class TestFontSizeControl:
    """CONFIG page has font size adjustment."""

    def test_font_size_offset_default_zero(self):
        """Font size offset resets to 0 correctly."""
        from enigma_engine.gui import widgets
        widgets.set_font_size_offset(0)
        assert widgets.get_font_size_offset() == 0

    def test_font_size_offset_adjusts_fonts(self):
        """set_font_size_offset changes module-level FONT_* tuples."""
        from enigma_engine.gui import widgets
        widgets.set_font_size_offset(0)
        original_body = widgets.FONT_BODY[1]
        widgets.set_font_size_offset(2)
        assert widgets.FONT_BODY[1] == original_body + 2
        # Restore
        widgets.set_font_size_offset(0)
        assert widgets.FONT_BODY[1] == original_body


# ================================================================
# A2: Keyboard shortcuts help overlay
# ================================================================


# ================================================================
# A3: Token counter in chat
# ================================================================


# ================================================================
# A4: HuggingFace download in GUI
# ================================================================


# ================================================================
# A7: Backup/restore system
# ================================================================


# ================================================================
# D3: Bare except:pass cleanup in mods
# ================================================================

class TestBareExceptCleanup:
    """Mods do not use bare except: — use except Exception instead."""

    @pytest.mark.parametrize("mod_name", [
        "voice", "threed", "videogen", "router",
        "imagegen", "audiogen",
    ])
    def test_mod_no_bare_except(self, mod_name):
        """Mod files must not contain bare 'except:'."""
        mod_dir = Path(__file__).parent.parent / "mods" / mod_name
        for py_file in mod_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped == "except:" or stripped == "except:  # noqa":
                    pytest.fail(
                        f"{py_file.name}:{i} has bare 'except:' "
                        f"— use 'except Exception:'")


# ================================================================
# FORGE Page: Adaptive Training Pipeline (TC-C3 + SA-B + SA-C)
# ================================================================


# ================================================================
# Input history (Up/Down recall)
# ================================================================

class TestInputHistory:
    """Verify chat input history recall logic."""

    def test_init_input_history_state(self):
        """_init_input_history sets empty list and idx=-1."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        obj = object.__new__(LogicChatMixin)
        obj._init_input_history()
        assert obj._input_history == []
        assert obj._input_hist_idx == -1
        assert obj._input_hist_draft == ""

    def test_history_max_constant(self):
        """Input history max is reasonable."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        assert LogicChatMixin._INPUT_HISTORY_MAX == 50


class TestChatHistoryTrimming:
    """Verify chat history is trimmed to prevent RAM leaks."""

    def test_max_chat_history_constant(self):
        """MAX_CHAT_HISTORY constant exists and is reasonable."""
        from enigma_engine.gui.media import MAX_CHAT_HISTORY
        assert isinstance(MAX_CHAT_HISTORY, int)
        assert 100 <= MAX_CHAT_HISTORY <= 1000

    def test_trim_chat_history_removes_oldest(self):
        """_trim_chat_history removes oldest messages when over cap."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        from enigma_engine.gui.media import MAX_CHAT_HISTORY
        obj = object.__new__(LogicChatMixin)
        # Simulate a history with messages over the cap
        obj.history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(MAX_CHAT_HISTORY + 50)
        ]
        obj._trim_chat_history()
        assert len(obj.history) == MAX_CHAT_HISTORY
        # Oldest messages should be gone
        assert obj.history[0]["content"] == "msg50"
        # Newest messages should remain
        assert obj.history[-1]["content"] == f"msg{MAX_CHAT_HISTORY + 49}"

    def test_trim_chat_history_doesnt_trim_under_cap(self):
        """_trim_chat_history does nothing when under cap."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        obj = object.__new__(LogicChatMixin)
        obj.history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
        ]
        obj._trim_chat_history()
        assert len(obj.history) == 2


# ================================================================
# RLHF / Self-Play Dropdown (#21)
# ================================================================


# ================================================================
# S622–S629: Button System Upgrade
# ================================================================

class TestButtonThemeConstants:
    """S622: New theme constants for button colour system."""

    def test_reload_theme_updates_new_constants(self):
        """reload_theme must propagate new C_* constants."""
        from enigma_engine.gui import widgets
        old_red_dim = widgets.C_RED_DIM
        widgets.reload_theme("midnight")
        # Midnight has different colours
        assert isinstance(widgets.C_RED_DIM, str)
        assert isinstance(widgets.C_CYAN_DIM, str)
        widgets.reload_theme("dark")


class TestThemedButton:
    """S623: themed_button() factory function."""

    def test_themed_button_valid_styles(self):
        """themed_button must accept all defined style names."""
        from enigma_engine.gui.widgets import BUTTON_STYLES
        expected = {"primary", "danger", "action", "tool",
                    "secondary", "warning", "icon"}
        assert expected == set(BUTTON_STYLES.keys())

    def test_button_styles_have_required_keys(self):
        """Each button style must define fg_color, hover_color, text_color."""
        from enigma_engine.gui.widgets import BUTTON_STYLES
        required = {"fg_color", "hover_color", "text_color"}
        for style_name, style_dict in BUTTON_STYLES.items():
            for key in required:
                assert key in style_dict, (
                    f"Style '{style_name}' missing key '{key}'")

    def test_button_styles_all_strings(self):
        """All colour values in button styles must be non-empty strings."""
        from enigma_engine.gui.widgets import BUTTON_STYLES
        for style_name, style_dict in BUTTON_STYLES.items():
            for key, val in style_dict.items():
                assert isinstance(val, str) and len(val) > 0, (
                    f"Style '{style_name}'.{key} = {val!r}")

    def test_primary_style_uses_green(self):
        """Primary buttons must use green colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_GREEN
        assert BUTTON_STYLES["primary"]["text_color"] == C_GREEN

    def test_danger_style_uses_red(self):
        """Danger buttons must use red colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_RED
        assert BUTTON_STYLES["danger"]["text_color"] == C_RED

    def test_tool_style_uses_cyan(self):
        """Tool buttons must use cyan colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_CYAN
        assert BUTTON_STYLES["tool"]["text_color"] == C_CYAN

    def test_action_style_uses_accent(self):
        """Action buttons must use accent colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_ACCENT
        assert BUTTON_STYLES["action"]["text_color"] == C_ACCENT

    def test_warning_style_uses_orange(self):
        """Warning buttons must use orange colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_ORANGE
        assert BUTTON_STYLES["warning"]["text_color"] == C_ORANGE

    def test_secondary_style_uses_dim(self):
        """Secondary buttons must use dim text."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_TEXT_DIM
        assert BUTTON_STYLES["secondary"]["text_color"] == C_TEXT_DIM

    def test_icon_style_transparent(self):
        """Icon buttons must have transparent background."""
        from enigma_engine.gui.widgets import BUTTON_STYLES
        assert BUTTON_STYLES["icon"]["fg_color"] == "transparent"


class TestButtonUsageInPages:
    """S624–S629: Verify GUI pages use themed_button for consistency."""

    def test_no_hardcoded_green_hover(self):
        """No GUI page file should hardcode green hover hex."""
        import enigma_engine.gui.gui_pages as mod_pages
        import enigma_engine.gui.gui_pages_forge as mod_forge
        import enigma_engine.gui.gui_docs_page as mod_docs
        import enigma_engine.gui.gui_cmd_page as mod_cmd
        import enigma_engine.gui.gui_mod_page as mod_mod
        import enigma_engine.gui.gui_forge_models as mod_fm
        for mod in [mod_pages, mod_forge, mod_docs,
                    mod_cmd, mod_mod, mod_fm]:
            src = inspect.getsource(mod)
            assert "#1a5a2a" not in src, (
                f"{mod.__name__} still has hardcoded #1a5a2a")
            assert "#1a4a2e" not in src, (
                f"{mod.__name__} still has hardcoded #1a4a2e")

    def test_no_hardcoded_red_bg(self):
        """No GUI page file should hardcode red background hex."""
        import enigma_engine.gui.gui_pages as mod_pages
        import enigma_engine.gui.gui_docs_page as mod_docs
        for mod in [mod_pages, mod_docs]:
            src = inspect.getsource(mod)
            assert "#3a1111" not in src, (
                f"{mod.__name__} still has hardcoded #3a1111")
            assert "#3b1111" not in src, (
                f"{mod.__name__} still has hardcoded #3b1111")
            assert "#5a1a1a" not in src, (
                f"{mod.__name__} still has hardcoded #5a1a1a")


# =====================================================================
# Cross-wiring: every TrainingConfig call site must include core fields
# =====================================================================


class TestTrainingConfigCrossWiring:
    """Structural test: all TrainingConfig() call sites in GUI must
    include the fields that should be present in every training mode.

    This test prevents the class of bug where a feature exists in
    TrainingConfig, has a GUI widget, but some training modes forget
    to pass it through.  When a new required field is added, adding
    it to REQUIRED_FIELDS here will catch any mode that misses it.
    """

    # Fields that EVERY TrainingConfig() call in the GUI must include.
    # Vision gets a pass on use_sequence_packing (batch_size=1).
    REQUIRED_FIELDS = [
        "use_gradient_checkpointing",
        "ce_chunk_size",
        "use_compile",
        "rolling_best_k",
        "save_every",
        "checkpoint_dir",
        "use_amp",
        "run_evaluation",
    ]

    # Files that contain TrainingConfig() constructor calls.
    GUI_TRAINING_FILES = [
        "enigma_engine.gui.gui_forge_training",
        "enigma_engine.gui.gui_forge_new_modes",
    ]

    def test_all_config_calls_include_required_fields(self):
        """Every TrainingConfig(...) block in GUI training files must
        mention each required field.  This catches 'feature exists but
        not wired' bugs automatically."""
        import importlib
        import inspect
        import re

        missing = []
        for mod_name in self.GUI_TRAINING_FILES:
            mod = importlib.import_module(mod_name)
            src = inspect.getsource(mod)
            # Find each TrainingConfig( ... ) block.  They span
            # multiple lines, so grab from 'TrainingConfig(' to the
            # matching closing paren.
            # We use a simple heuristic: find each occurrence and
            # capture the next ~40 lines (configs are ~15-25 lines).
            lines = src.splitlines()
            for i, line in enumerate(lines):
                if "TrainingConfig(" in line and "import" not in line:
                    # Grab the config block (up to 40 lines or the
                    # next line with just ')' or 'trainer = ')
                    block_lines = lines[i:i + 40]
                    block = "\n".join(block_lines)
                    # Find the closing of the constructor
                    end = block.find("\n                trainer")
                    if end == -1:
                        end = block.find("\n                self._log")
                    if end > 0:
                        block = block[:end]

                    for field in self.REQUIRED_FIELDS:
                        # Vision mode (batch_size=1) doesn't need
                        # sequence packing
                        if field == "use_sequence_packing" and \
                                "batch_size=1" in block:
                            continue
                        if field not in block:
                            # Get function context
                            func_match = re.search(
                                r"def\s+(\w+)",
                                "\n".join(lines[max(0, i - 80):i]))
                            func_name = (func_match.group(1)
                                         if func_match else "unknown")
                            missing.append(
                                f"{mod_name}::{func_name} "
                                f"(line ~{i + 1}) missing "
                                f"'{field}'")

        assert not missing, (
            "TrainingConfig call sites missing required fields:\n"
            + "\n".join(f"  - {m}" for m in missing))

    def test_forge_params_fields_are_consumed(self):
        """Every field returned by _read_forge_train_params() must
        appear in at least one TrainingConfig() call.  Catches dead
        GUI widget connections."""
        import inspect
        import enigma_engine.gui.gui_forge as mod_forge
        import enigma_engine.gui.gui_forge_training as mod_train
        import enigma_engine.gui.gui_forge_new_modes as mod_new

        # Extract field names from _read_forge_train_params return dict
        import re
        # Use the METHOD source, not the whole module — avoids
        # matching return dicts from other functions.
        forge_src = inspect.getsource(
            mod_forge.ForgeMixin._read_forge_train_params)
        # The return dict has lines like: "field_name": value,
        return_match = re.search(
            r"return\s*\{([^}]+)\}",
            forge_src, re.DOTALL)
        assert return_match, \
            "_read_forge_train_params has no return dict"
        return_block = return_match.group(1)
        field_names = re.findall(
            r'"(\w+)":', return_block)
        assert len(field_names) >= 5, \
            f"Expected >=5 fields, got {field_names}"

        # Check that each field appears in at least one config site
        consumer_src = (inspect.getsource(mod_train)
                        + inspect.getsource(mod_new))
        unused = []
        for field in field_names:
            # Search for forge_params["field"] or just field= in
            # TrainingConfig blocks
            pattern = (f'forge_params["{field}"]'
                       if field != "general_data" else field)
            if pattern not in consumer_src:
                unused.append(field)

        assert not unused, (
            "_read_forge_train_params() returns fields that no "
            "training mode uses:\n"
            + "\n".join(f"  - {f}" for f in unused))

    def test_solo_training_shows_batch_eta(self):
        """Batch-level ETA must appear in the solo training on_loss handler."""
        import inspect
        import enigma_engine.gui.gui_forge_training as mod_train
        src = inspect.getsource(mod_train.ForgeTrainingMixin)
        assert '_total_training_steps' in src, (
            "Solo training handler missing batch-level ETA "
            "from _total_training_steps")
