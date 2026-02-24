"""Tests for the desktop GUI module and widgets."""

import json
import inspect
from pathlib import Path

import pytest


# ================================================================
# Scanners
# ================================================================

class TestScanners:
    """Verify all filesystem scanners work."""

    def test_scan_bricks(self):
        from enigma_engine.gui.scanners import scan_bricks
        bricks = scan_bricks()
        ids = [b["id"] for b in bricks]
        assert "echo" in ids
        assert "imagegen" in ids
        assert "_template" not in ids
        for brick in bricks:
            assert "id" in brick
            assert "name" in brick
            assert "commands" in brick
            assert "prompt" in brick

    def test_scan_models(self):
        from enigma_engine.gui.scanners import scan_models
        models = scan_models()
        assert isinstance(models, list)
        for m in models:
            assert "name" in m
            assert "path" in m
            assert "size_mb" in m

    def test_scan_profiles(self):
        from enigma_engine.gui.scanners import scan_profiles
        profiles = scan_profiles()
        assert isinstance(profiles, list)
        ids = [p["id"] for p in profiles]
        assert "assistant" in ids
        for p in profiles:
            assert "name" in p
            assert "description" in p

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
        """scan_docs returns guides, profiles, and brick docs."""
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
        # Has profiles and brick docs
        assert any(d["category"] == "profiles" for d in docs)
        assert any(d["category"].startswith("brick:") for d in docs)
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
        assert clamp_config("max_tokens", 0) == 16
        assert clamp_config("max_tokens", 99999) == 4096

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

    def test_scanners_module(self):
        from enigma_engine.gui import scanners
        for attr in ("scan_bricks", "scan_models", "scan_profiles",
                     "scan_training_data", "scan_sessions", "scan_docs",
                     "clamp_config", "CONFIG_LIMITS",
                     "CONFIG_DESCRIPTIONS", "ROUTE_KEYS",
                     "PATH_SETTINGS", "load_path_settings",
                     "save_path_settings", "get_path",
                     "OUTPUTS_DIR", "INFO_DIR"):
            assert hasattr(scanners, attr), f"Missing: {attr}"

    def test_all_mixins_exist(self):
        from enigma_engine.gui.gui_pages import PagesMixin
        from enigma_engine.gui.gui_logic import LogicMixin
        from enigma_engine.gui.gui_brick_page import BrickPageMixin
        from enigma_engine.gui.gui_forge import ForgeMixin
        from enigma_engine.gui.gui_bricks import BrickMixin
        from enigma_engine.gui.gui_cmd_page import CMDPageMixin
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        # Page builders
        assert hasattr(PagesMixin, "_build_page_core")
        assert hasattr(PagesMixin, "_build_page_config")
        assert hasattr(BrickPageMixin, "_build_page_brick")
        assert hasattr(CMDPageMixin, "_build_page_cmd")
        assert hasattr(DocsPageMixin, "_build_page_docs")
        # Core logic
        assert hasattr(LogicMixin, "_send_message")
        assert hasattr(LogicMixin, "_load_model")
        # Forge logic
        assert hasattr(ForgeMixin, "_start_model_training")
        assert hasattr(ForgeMixin, "_model_config_dict")
        # Brick logic
        assert hasattr(BrickMixin, "_toggle_brick")
        assert hasattr(BrickMixin, "_launch_brick")

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
            scan_bricks, scan_models, scan_profiles, scan_docs,
            INFO_DIR, clamp_config, CONFIG_LIMITS)
        assert callable(scan_bricks)
        assert callable(scan_docs)
        assert isinstance(CONFIG_LIMITS, dict)
        assert INFO_DIR.exists()


# ================================================================
# Brick template completeness
# ================================================================

class TestBrickTemplate:
    """Verify _template has everything for GUI connection."""

    def test_template_complete(self):
        import json as _json
        from enigma_engine.gui.scanners import BRICKS_DIR
        tpl = BRICKS_DIR / "_template" / "brick.json"
        data = _json.loads(tpl.read_text(encoding="utf-8"))
        assert isinstance(data.get("prompt"), str)
        assert len(data["prompt"]) > 0
        assert "widgets" in data.get("ui", {})
        assert len(data["ui"]["widgets"]) > 0
        assert len(data.get("commands", [])) > 0
        for cmd in data["commands"]:
            assert "name" in cmd
            assert "description" in cmd


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

    def test_cmd_uses_subprocess(self):
        """CMD page imports subprocess for real shell."""
        import importlib
        mod = importlib.import_module(
            "enigma_engine.gui.gui_cmd_page")
        source = open(mod.__file__).read()
        assert "import subprocess" in source
        assert "powershell" in source.lower()

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

    def test_logic_has_context_methods(self):
        from enigma_engine.gui.gui_logic import LogicMixin
        for attr in ("_save_model_context",
                     "_load_model_context",
                     "_restore_history_display"):
            assert hasattr(LogicMixin, attr), f"Missing: {attr}"


# ================================================================
# CORE page widgets
# ================================================================

class TestCorePage:
    """Verify CORE page has all expected widgets and features."""

    def test_collapsible_sidebar(self):
        """CORE page uses CollapsiblePanel for sidebar sections."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "CollapsiblePanel" in source
        assert "_sidebar_toggle_btn" in source
        assert "_toggle_sidebar" in source
        assert hasattr(PagesMixin, "_toggle_sidebar")

    def test_toolbar_layout(self):
        """SEND is separate from utility buttons in toolbar row."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "toolbar" in source
        assert "send_btn" in source
        assert "_new_btn" in source
        assert "Start new conversation" in source
        assert "Tooltip" in source

    def test_selectable_text_in_pages(self):
        """Pages use SelectableTextbox for output displays."""
        from enigma_engine.gui.gui_pages import PagesMixin
        from enigma_engine.gui.gui_cmd_page import CMDPageMixin
        from enigma_engine.gui.gui_brick_page import BrickPageMixin
        assert "SelectableTextbox" in inspect.getsource(PagesMixin)
        assert "SelectableTextbox" in inspect.getsource(
            CMDPageMixin._build_page_cmd)
        assert "SelectableTextbox" in inspect.getsource(
            BrickPageMixin._build_page_brick)

    def test_label_copy(self):
        """All labels support right-click copy."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_enable_label_copy")
        source = inspect.getsource(EnigmaGUI._enable_label_copy)
        assert "winfo_children" in source
        assert "CTkLabel" in source
        assert "_enable_label_copy" in inspect.getsource(
            EnigmaGUI.__init__)


# ================================================================
# Voice input
# ================================================================

class TestVoiceInput:
    """Voice input is non-blocking and stoppable."""

    def test_voice_methods(self):
        from enigma_engine.gui.gui_logic import LogicMixin
        for attr in ("_toggle_voice_input", "_on_voice_text",
                     "_voice_input_done"):
            assert hasattr(LogicMixin, attr), f"Missing: {attr}"

    def test_voice_uses_background_listener(self):
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._toggle_voice_input)
        assert "listen_in_background" in source
        assert "_voice_stopper" in source


# ================================================================
# FORGE data editor
# ================================================================

class TestDataEditor:
    """Verify data editor on FORGE page."""

    def test_forge_data_editor(self):
        from enigma_engine.gui.gui_forge import ForgeMixin
        from enigma_engine.gui.gui_pages import PagesMixin
        for attr in ("_load_data_into_editor", "_save_data_file",
                     "_new_data_file", "_refresh_data_files"):
            assert hasattr(ForgeMixin, attr), f"Missing: {attr}"
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "data_editor" in source
        assert "_load_data_into_editor" in source


# ================================================================
# Nav rail
# ================================================================

class TestNavRail:
    """Verify collapsible nav rail and shell structure."""

    def test_nav_toggle(self):
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_toggle_nav")
        source = inspect.getsource(EnigmaGUI._build_shell)
        assert "_nav_toggle" in source
        assert "Tooltip" in source
        assert '"DOCS"' in source

    def test_nav_collapse_behavior(self):
        """Nav toggle hides nav via grid_remove."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._toggle_nav)
        assert "grid_remove" in source
        assert "grid_columnconfigure" in source


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

    def test_config_page_has_paths(self):
        from enigma_engine.gui.gui_pages import PagesMixin
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(PagesMixin._build_page_config)
        assert "path_entries" in source
        assert "DIRECTORY PATHS" in source
        for attr in ("_browse_path", "_save_paths", "_reset_paths"):
            assert hasattr(LogicMixin, attr), f"Missing: {attr}"


# ================================================================
# DOCS page
# ================================================================

class TestDocsPage:
    """DOCS page: documentation browser, profiles, file management."""

    def test_docs_mixin_methods(self):
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        for method in (
            "_build_page_docs", "_docs_open", "_docs_save",
            "_docs_delete", "_docs_refresh",
            "_docs_new_file", "_docs_new_profile",
        ):
            assert hasattr(DocsPageMixin, method), f"Missing {method}"

    def test_docs_wired_into_desktop(self):
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_build_page_docs" in source


# ================================================================
# Chat fullscreen
# ================================================================

class TestChatFullscreen:
    """Verify chat fullscreen covers the entire GUI."""

    def test_fullscreen_toggle_method(self):
        """PagesMixin has the fullscreen toggle method."""
        from enigma_engine.gui.gui_pages import PagesMixin
        assert hasattr(PagesMixin, "_toggle_chat_fullscreen")
        assert hasattr(PagesMixin, "_exit_chat_fullscreen")

    def test_fullscreen_button_in_core(self):
        """CORE page has a fullscreen toggle button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "_fullscreen_btn" in source
        assert "_toggle_chat_fullscreen" in source

    def test_fullscreen_state_attribute(self):
        """Desktop GUI has fullscreen state attribute."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_chat_fullscreen" in source

    def test_header_ref_stored(self):
        """Header frame is stored for fullscreen toggling."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._build_shell)
        assert "_header" in source

    def test_escape_exits_fullscreen(self):
        """Escape key binding exits fullscreen mode."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._toggle_chat_fullscreen)
        assert "Escape" in source


# ================================================================
# Display names and model AI name
# ================================================================

class TestDisplayNames:
    """Verify configurable user/AI names in chat."""

    def test_name_attributes_on_gui(self):
        """EnigmaGUI has user_name and ai_name attributes."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "user_name" in source
        assert "ai_name" in source

    def test_names_used_in_chat(self):
        """Chat messages use configurable names, not hardcoded."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "self.user_name" in source
        assert "\"YOU\"" not in source

    def test_names_used_in_history_restore(self):
        """History restore uses configurable names."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._restore_history_display)
        assert "self.user_name" in source
        assert "_active_ai_name" in source

    def test_model_display_name_method(self):
        """Logic mixin has model display name loading."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_load_model_display_name")
        assert hasattr(LogicMixin, "_active_ai_name")

    def test_model_info_json(self):
        """Qwen model folder has a model_info.json."""
        from enigma_engine.gui.scanners import MODELS_DIR
        info = MODELS_DIR / "qwen2.5-14b-instruct" / "model_info.json"
        assert info.exists()
        data = json.loads(info.read_text(encoding="utf-8"))
        assert "display_name" in data
        assert len(data["display_name"]) > 0

    def test_name_entries_in_config(self):
        """CONFIG page has display name entries."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_config)
        assert "_user_name_entry" in source
        assert "_ai_name_entry" in source
        assert "DISPLAY NAMES" in source
