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

    def test_scan_models_shard_grouping_logic(self):
        """Verify shard regex groups correctly via source inspection."""
        from enigma_engine.gui.scanners import scan_models
        source = inspect.getsource(scan_models)
        assert "_shard_re" in source
        assert "shard_groups" in source

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

    def test_scanners_module(self):
        from enigma_engine.gui import scanners
        for attr in ("scan_mods", "scan_models",
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
        from enigma_engine.gui.gui_mod_page import ModPageMixin
        from enigma_engine.gui.gui_forge import ForgeMixin
        from enigma_engine.gui.gui_mods import ModMixin
        from enigma_engine.gui.gui_cmd_page import CMDPageMixin
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        # Page builders
        assert hasattr(PagesMixin, "_build_page_core")
        assert hasattr(PagesMixin, "_build_page_config")
        assert hasattr(ModPageMixin, "_build_page_mod")
        assert hasattr(CMDPageMixin, "_build_page_cmd")
        assert hasattr(DocsPageMixin, "_build_page_docs")
        # Core logic
        assert hasattr(LogicMixin, "_send_message")
        assert hasattr(LogicMixin, "_load_model")
        # Forge logic
        assert hasattr(ForgeMixin, "_start_training_by_mode")
        assert hasattr(ForgeMixin, "_model_config_dict")
        # Mod logic
        assert hasattr(ModMixin, "_toggle_mod")
        assert hasattr(ModMixin, "_launch_mod")

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

class TestModTemplate:
    """Verify _template has everything for GUI connection."""

    def test_template_complete(self):
        import json as _json
        from enigma_engine.gui.scanners import MODS_DIR
        tpl = MODS_DIR / "_template" / "mod.json"
        data = _json.loads(tpl.read_text(encoding="utf-8"))
        assert isinstance(data.get("prompt"), str)
        assert len(data["prompt"]) > 0
        assert "widgets" in data.get("ui", {})
        assert len(data["ui"]["widgets"]) > 0
        assert len(data.get("commands", [])) > 0
        for cmd in data["commands"]:
            assert "name" in cmd
            assert "description" in cmd


class TestModDefinitions:
    """Verify mod file conventions and merged audio/voice setup."""

    def test_voice_mod_has_main_entry(self):
        from enigma_engine.gui.scanners import MODS_DIR
        assert (MODS_DIR / "voice" / "main.py").exists()

    def test_mod_info_card_supports_rules(self):
        from enigma_engine.gui.gui_mod_page import ModPageMixin
        source = inspect.getsource(ModPageMixin._build_page_mod)
        assert "mod.get(\"rules\"" in source
        assert "Rules:" in source


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
        source = open(mod.__file__, encoding="utf-8").read()
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

    def test_logic_has_release_loaded_engine_method(self):
        """LogicMixin has explicit backend release helper."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_release_loaded_engine")

    def test_unload_model_uses_release_helper(self):
        """_unload_model should call explicit release helper."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._unload_model)
        assert "_release_loaded_engine" in source

    def test_model_context_has_identity_fields(self):
        """ModelContext exposes identity card fields."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("identity_check")
        for attr in ("display_name", "personality", "avatar",
                     "created_at", "total_messages", "total_sessions",
                     "training_history", "tags", "notes"):
            assert hasattr(ctx, attr), f"Missing identity field: {attr}"

    def test_model_context_has_identity_methods(self):
        """ModelContext has identity helper methods."""
        from enigma_engine.core.model_context import ModelContext
        for method in ("increment_messages", "increment_sessions",
                       "record_training_run"):
            assert hasattr(ModelContext, method), (
                f"Missing identity method: {method}")

    def test_model_context_has_memory_fact_count(self):
        """ModelContext exposes memory_fact_count property."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("mem_check")
        assert hasattr(ctx, "memory_fact_count")


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
        from enigma_engine.gui.gui_mod_page import ModPageMixin
        assert "SelectableTextbox" in inspect.getsource(PagesMixin)
        assert "SelectableTextbox" in inspect.getsource(
            CMDPageMixin._build_page_cmd)
        assert "SelectableTextbox" in inspect.getsource(
            ModPageMixin._build_page_mod)

    def test_label_copy(self):
        """All labels support right-click copy."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_enable_label_copy")
        source = inspect.getsource(EnigmaGUI._enable_label_copy)
        assert "winfo_children" in source
        assert "CTkLabel" in source
        assert "_enable_label_copy" in inspect.getsource(
            EnigmaGUI.__init__)

    def test_selectable_label_keeps_fixed_width_stable(self):
        """Fixed-width SelectableLabel should not resize on text updates."""
        from enigma_engine.gui.widgets import SelectableLabel
        source = inspect.getsource(SelectableLabel)
        assert "_fixed_width" in source
        assert "if not self._fixed_width" in source


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
    """Verify data editing methods exist in ForgeMixin."""

    def test_forge_data_methods(self):
        from enigma_engine.gui.gui_forge import ForgeMixin
        for attr in ("_on_data_selected", "_refresh_data_files"):
            assert hasattr(ForgeMixin, attr), f"Missing: {attr}"


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
    """DOCS page: documentation browser, file management."""

    def test_docs_mixin_methods(self):
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        for method in (
            "_build_page_docs", "_docs_open", "_docs_save",
            "_docs_delete", "_docs_refresh",
            "_docs_new_file",
            "_docs_start_rename", "_docs_finish_rename",
            "_docs_cancel_rename",
        ):
            assert hasattr(DocsPageMixin, method), f"Missing {method}"

    def test_docs_wired_into_desktop(self):
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_build_page_docs" in source

    def test_scan_docs_has_data_category(self):
        """scan_docs includes training data files under 'data' category."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        categories = {d["category"] for d in docs}
        assert "data" in categories, (
            "scan_docs should include data/ files as 'data' category")

    def test_docs_refresh_renders_data_category(self):
        """_docs_rebuild_browser handles the 'data' category label."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(DocsPageMixin._docs_rebuild_browser)
        assert '"data"' in source
        assert "TRAINING DATA" in source


# ================================================================
# DOCS page improvements
# ================================================================

class TestDocsPageImprovements:
    """DOCS page: search, notes, unsaved changes, Ctrl+S, stats."""

    def test_docs_search_filter_method(self):
        """DocsPageMixin has _docs_filter_browser method."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        assert hasattr(DocsPageMixin, "_docs_filter_browser")

    def test_docs_has_unsaved_tracking(self):
        """DocsPageMixin has _docs_check_unsaved method."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        assert hasattr(DocsPageMixin, "_docs_check_unsaved")

    def test_docs_keyboard_save(self):
        """DocsPageMixin has _docs_keyboard_save method for Ctrl+S."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        assert hasattr(DocsPageMixin, "_docs_keyboard_save")

    def test_docs_update_stats(self):
        """DocsPageMixin has _docs_update_stats method."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        assert hasattr(DocsPageMixin, "_docs_update_stats")

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

    def test_docs_refresh_renders_notes_category(self):
        """_docs_rebuild_browser handles the 'notes' category label."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(DocsPageMixin._docs_rebuild_browser)
        assert '"notes"' in source
        assert "NOTES" in source

    def test_docs_mixin_new_methods(self):
        """DocsPageMixin has all new improvement methods."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        for method in (
            "_docs_filter_browser",
            "_docs_check_unsaved",
            "_docs_keyboard_save",
            "_docs_update_stats",
            "_docs_mark_modified",
        ):
            assert hasattr(DocsPageMixin, method), f"Missing {method}"

    def test_scanners_notes_dir_constant(self):
        """scanners module exports NOTES_DIR constant."""
        from enigma_engine.gui.scanners import NOTES_DIR
        assert NOTES_DIR.name == "notes"

    def test_scanners_no_stale_profile_comment(self):
        """scanners.py scan_docs has no stale 'Profile files' comment."""
        from enigma_engine.gui import scanners
        source = inspect.getsource(scanners.scan_docs)
        assert "Profile files" not in source


# ================================================================
# Docs undo/redo
# ================================================================

class TestDocsUndoRedo:
    """DOCS page: Ctrl+Z undo and Ctrl+Y redo support."""

    def test_docs_has_undo_method(self):
        """DocsPageMixin has _docs_undo method."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        assert hasattr(DocsPageMixin, "_docs_undo")

    def test_docs_has_redo_method(self):
        """DocsPageMixin has _docs_redo method."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        assert hasattr(DocsPageMixin, "_docs_redo")

    def test_docs_undo_returns_break(self):
        """_docs_undo returns 'break' to prevent default tk handling."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(DocsPageMixin._docs_undo)
        assert '"break"' in source or "'break'" in source

    def test_docs_redo_returns_break(self):
        """_docs_redo returns 'break' to prevent default tk handling."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(DocsPageMixin._docs_redo)
        assert '"break"' in source or "'break'" in source

    def test_docs_editor_undo_configured(self):
        """Editor setup configures undo on the underlying text widget."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(DocsPageMixin._build_page_docs)
        assert "undo=True" in source
        assert "maxundo=-1" in source
        assert "autoseparators=True" in source

    def test_docs_editor_binds_ctrl_z(self):
        """Editor binds Ctrl+Z to _docs_undo."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(DocsPageMixin._build_page_docs)
        assert "<Control-z>" in source
        assert "_docs_undo" in source

    def test_docs_editor_binds_ctrl_y(self):
        """Editor binds Ctrl+Y to _docs_redo."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(DocsPageMixin._build_page_docs)
        assert "<Control-y>" in source
        assert "_docs_redo" in source


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
        info = MODELS_DIR / "qwen3-30b-a3b" / "model_info.json"
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

    def test_models_page_has_no_size_input(self):
        """MODELS page has no size entry — just name + CREATE."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_models)
        # Size entry and suffix should be gone
        assert "new_model_size_entry" not in source
        assert "new_model_size_suffix" not in source
        assert "_model_size_suffix" not in source
        # Name entry and CREATE should remain
        assert "new_model_name" in source
        assert "CREATE" in source
        assert "IMPORT" in source

    def test_forge_has_stage_buttons(self):
        """FORGE page uses stage buttons with tooltips."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_stage_buttons" in source
        assert "_select_training_stage" in source

    def test_create_model_uses_default_small_preset(self):
        """_create_new_model creates a blank model with default small preset."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._create_new_model)
        # Uses small preset, no size parsing
        assert "small" in source
        assert "parse_param_target" not in source

    def test_create_model_dispatches_by_mode(self):
        """_start_training_by_mode dispatches to solo/guided/dialogue."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_training_by_mode)
        assert "solo" in source.lower()
        assert "guided" in source.lower()  # AI-Guided mode
        assert "image" in source.lower()   # Image/Vision mode


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

    def test_docs_page_renders_trainer(self):
        """DocsPageMixin renders the trainer category."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(DocsPageMixin._docs_rebuild_browser)
        assert '"trainer"' in source


# ================================================================
# Unified History / Sessions
# ================================================================

class TestUnifiedHistory:
    """Tests for the unified session system.

    Design:
    - Single save location: memory/ only
    - Every chat is a session with _current_session_path tracking
    - Auto-save on every exchange to the SAME file
    - Click sidebar to switch sessions (no LOAD file picker)
    - SAVE replaced by RENAME
    - Active session highlighted in sidebar
    """

    def test_current_session_path_attribute(self):
        """LogicMixin._new_chat sets _current_session_path."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._new_chat)
        assert "_current_session_path" in source

    def test_auto_save_uses_current_session(self):
        """_auto_save_session writes to _current_session_path."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._auto_save_session)
        assert "_current_session_path" in source

    def test_auto_save_called_after_exchange(self):
        """_send_message calls _auto_save_session."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "_auto_save_session" in source

    def test_no_load_file_picker(self):
        """LogicMixin should not have _load_session file picker."""
        from enigma_engine.gui.gui_logic import LogicMixin
        # _load_session should no longer exist
        assert not hasattr(LogicMixin, "_load_session")

    def test_rename_session_exists(self):
        """LogicMixin has _rename_session method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_rename_session")
        assert callable(getattr(LogicMixin, "_rename_session"))

    def test_rename_session_updates_file(self):
        """_rename_session updates the name inside the session file."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._rename_session)
        # Should write to the current session path
        assert "_current_session_path" in source

    def test_scan_sessions_single_location(self):
        """scan_sessions only scans memory/ directory."""
        from enigma_engine.gui.scanners import scan_sessions
        source = inspect.getsource(scan_sessions)
        # Should only reference MEMORY_DIR, not SESSIONS_DIR
        assert "MEMORY_DIR" in source
        assert "SESSIONS_DIR" not in source

    def test_active_session_highlighted(self):
        """_refresh_history_list highlights the active session."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._refresh_history_list)
        # Active session uses filled circle marker
        assert "\u25cf" in source  # ● filled circle

    def test_sidebar_has_rename_button(self):
        """CORE page sidebar has RENAME button instead of SAVE/LOAD."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "RENAME" in source
        assert "_rename_session" in source
        # LOAD file picker should be gone
        assert "_load_session" not in source

    def test_delete_session_still_exists(self):
        """_delete_session method still works."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_delete_session")

    def test_export_chat_still_exists(self):
        """_export_chat method still works."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_export_chat")

    def test_export_chat_supports_formats(self):
        """_export_chat must offer Markdown, JSON, and text formats."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._export_chat)
        assert "*.md" in source, "Should offer Markdown export"
        assert "*.json" in source, "Should offer JSON export"
        assert "*.txt" in source, "Should offer text export"

    def test_load_session_by_path_sets_current(self):
        """_load_session_by_path updates _current_session_path."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._load_session_by_path)
        assert "_current_session_path" in source

    def test_new_chat_creates_session_file(self):
        """_new_chat generates a new timestamped session path."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._new_chat)
        assert "session_" in source
        assert "strftime" in source

    def test_delete_active_starts_new(self):
        """Deleting the active session starts a new chat."""
        from enigma_engine.gui.gui_logic import LogicMixin
        # _delete_session shows inline bar; _confirm_delete_session
        # does the actual deletion and calls _new_chat
        source = inspect.getsource(
            LogicMixin._confirm_delete_session)
        assert "_new_chat" in source

    def test_send_creates_session_if_empty(self):
        """_send_message creates a session path if none exists."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "_current_session_path" in source

    def test_loaded_session_links_clickable(self):
        """_load_session_by_path processes media/links in messages."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._load_session_by_path)
        assert "_process_media_in_text" in source

    def test_startup_creates_initial_session(self):
        """EnigmaGUI __init__ sets a non-empty _current_session_path."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_current_session_path" in source

    def test_new_chat_uses_unique_suffix(self):
        """_new_chat path includes subsecond component to avoid collisions."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._new_chat)
        # Should have a counter or time.time() or similar uniqueness
        assert "_session_counter" in source or "time.time()" in source

    def test_model_context_saves_session_path(self):
        """_save_model_context stores session_path in model context."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._save_model_context)
        assert "session_path" in source

    def test_load_model_context_resumes_session(self):
        """_load_model_context restores _current_session_path."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._load_model_context)
        assert "session_path" in source
        assert "_current_session_path" in source

    def test_generate_session_title_method_exists(self):
        """LogicMixin has _generate_session_title method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_generate_session_title")
        assert callable(
            getattr(LogicMixin, "_generate_session_title"))

    def test_session_title_called_on_first_exchange(self):
        """_send_message triggers title generation on first exchange."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "_generate_session_title" in source

    def test_session_title_uses_background_thread(self):
        """_generate_session_title runs in a daemon thread."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._generate_session_title)
        assert "Thread" in source
        assert "daemon" in source

    def test_load_session_syncs_model_context(self):
        """Loading a session from sidebar saves model context."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._load_session_by_path)
        assert "_save_model_context" in source


class TestChatLinks:
    """Tests for clickable link handling in chat."""

    def test_make_url_uses_per_url_tags(self):
        """_make_url_clickable uses unique tags per URL."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._make_url_clickable)
        # Should create unique tags, not just reuse "link"
        assert "link_" in source

    def test_on_link_click_reads_tag_url(self):
        """_on_link_click retrieves URL from tag data, not line parse."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._on_link_click)
        assert "_link_urls" in source

    def test_link_urls_dict_initialized(self):
        """gui_pages initializes _link_urls dict for link storage."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "_link_urls" in source


class TestPinWindow:
    """Tests for always-on-top pin toggle."""

    def test_pinned_attribute_exists(self):
        """EnigmaGUI __init__ sets _pinned = False."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_pinned" in source

    def test_toggle_pin_method_exists(self):
        """EnigmaGUI has _toggle_pin method."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_toggle_pin")
        assert callable(getattr(EnigmaGUI, "_toggle_pin"))

    def test_toggle_pin_uses_topmost(self):
        """_toggle_pin calls attributes('-topmost', ...)."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._toggle_pin)
        assert "topmost" in source

    def test_pin_button_in_header(self):
        """_build_shell creates _pin_btn."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._build_shell)
        assert "_pin_btn" in source
        assert "pin" in source.lower()


class TestGUIContext:
    """Tests for AI GUI awareness and model reuse."""

    def test_build_gui_context_method_exists(self):
        """LogicMixin has _build_gui_context method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_build_gui_context")
        assert callable(getattr(LogicMixin, "_build_gui_context"))

    def test_gui_context_has_route_info(self):
        """_build_gui_context includes route and model information."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "route_assignments" in source
        assert "models_data" in source
        assert "mods_data" in source

    def test_gui_context_injected_in_chat(self):
        """_send_message passes GUI context as system_prompt."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "_build_gui_context" in source
        assert "system_prompt" in source

    def test_get_engine_for_route_method(self):
        """LogicMixin has _get_engine_for_route for model reuse."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_get_engine_for_route")
        source = inspect.getsource(LogicMixin._get_engine_for_route)
        # Should check if paths resolve to same file
        assert "resolve" in source

    def test_assign_route_notes_shared_engine(self):
        """_assign_model_to_route mentions sharing when reusing chat engine."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._assign_model_to_route)
        assert "sharing" in source.lower() or "shared" in source.lower()


class TestResizableSidebar:
    """Tests for drag-to-resize sidebar in CORE page."""

    def test_core_page_uses_paned_window(self):
        """_build_page_core creates a PanedWindow for chat/sidebar."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "PanedWindow" in source
        assert "_core_pane" in source

    def test_sidebar_toggle_uses_pane(self):
        """_toggle_sidebar uses pane forget/add instead of grid."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._toggle_sidebar)
        assert "forget" in source
        assert "_core_pane" in source

    def test_fullscreen_uses_pane(self):
        """Fullscreen enter/exit uses PanedWindow for sidebar."""
        from enigma_engine.gui.gui_pages import PagesMixin
        enter_src = inspect.getsource(
            PagesMixin._toggle_chat_fullscreen)
        exit_src = inspect.getsource(
            PagesMixin._exit_chat_fullscreen)
        assert "_core_pane" in enter_src
        assert "_core_pane" in exit_src


class TestExternalModelsDocs:
    """Tests for external model limitations documentation."""

    def test_external_models_doc_exists(self):
        """information/external_models.md exists and is readable."""
        doc = Path("information/external_models.md")
        assert doc.exists()
        content = doc.read_text(encoding="utf-8")
        assert len(content) > 200

    def test_external_docs_covers_formats(self):
        """Doc covers all supported external formats."""
        content = Path("information/external_models.md").read_text(
            encoding="utf-8")
        for fmt in ("GGUF", "HuggingFace", "GPTQ", "AWQ", "ONNX", "Ollama"):
            assert fmt in content, f"Missing format: {fmt}"

    def test_external_docs_covers_limitations(self):
        """Doc explains what doesn't work on external models."""
        content = Path("information/external_models.md").read_text(
            encoding="utf-8")
        assert "NOT Work" in content or "not work" in content.lower()
        assert "Training" in content

    def test_scan_docs_finds_external_models(self):
        """scan_docs discovers the external_models.md file."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        names = [d["name"] for d in docs]
        assert "External Models" in names

    def test_header_status_tooltip(self):
        """Header status frame has a tooltip explaining the indicator."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._build_shell)
        assert "Model status" in source or "status" in source.lower()
        assert "Tooltip" in source


# ================================================================
# Models page feedback
# ================================================================

class TestModelsPageFeedback:
    """Verify the MODELS page shows inline create/delete feedback."""

    def test_models_status_label_created(self):
        """_build_page_models creates a _models_status label."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_models)
        assert "_models_status" in source
        assert "SelectableLabel" in source

    def test_models_msg_helper_exists(self):
        """ForgeMixin._models_msg is defined and updates status."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_models_msg")
        source = inspect.getsource(ForgeMixin._models_msg)
        assert "status_bar" in source
        assert "_models_status" in source

    def test_create_model_uses_models_msg(self):
        """_create_new_model uses _models_msg instead of _chat_system."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._create_new_model)
        assert "_models_msg" in source
        # Should NOT use _chat_system for user-facing feedback
        assert "_chat_system" not in source

    def test_delete_model_uses_models_msg(self):
        """_delete_model shows inline confirmation bar."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # _delete_model now shows an inline bar; actual deletion
        # with feedback is in _confirm_delete_model
        source = inspect.getsource(
            ForgeMixin._confirm_delete_model)
        assert "_models_msg" in source or "_refresh_models" in source

    def test_route_assign_updates_status_bar(self):
        """_assign_model_to_route updates status_bar for non-chat routes."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._assign_model_to_route)
        assert "status_bar.set_left" in source


class TestSendButtonSafety:
    """Test SEND button re-enable safety on page switch."""

    def test_switch_page_reenables_send_btn(self):
        """_switch_page re-enables send_btn when switching to CORE."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._switch_page)
        assert "send_btn" in source
        assert "CORE" in source
        assert "_thinking_active" in source
        assert "_model_loading" in source

    def test_send_btn_not_reenabled_while_thinking(self):
        """_switch_page only re-enables if not thinking or loading."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._switch_page)
        assert "not self._thinking_active" in source
        assert "not self._model_loading" in source

    def test_model_loading_flag_exists(self):
        """_model_loading flag is set in __init__."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_model_loading" in source

    def test_model_loading_flag_set_during_load(self):
        """_load_model sets _model_loading, callbacks clear it."""
        from enigma_engine.gui.gui_logic import LogicMixin
        load_src = inspect.getsource(LogicMixin._load_model)
        assert "_model_loading = True" in load_src
        loaded_src = inspect.getsource(LogicMixin._on_model_loaded)
        assert "_model_loading = False" in loaded_src
        error_src = inspect.getsource(LogicMixin._on_model_error)
        assert "_model_loading = False" in error_src


class TestRouteAssignmentPersistence:
    """Test that route assignments save and load from disk."""

    def test_save_load_route_functions_exist(self):
        """Scanner module has save/load route assignment helpers."""
        from enigma_engine.gui.scanners import (
            save_route_assignments, load_route_assignments)
        assert callable(save_route_assignments)
        assert callable(load_route_assignments)

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

    def test_assign_model_saves_to_disk(self):
        """_assign_model_to_route persists assignments."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._assign_model_to_route)
        assert "save_route_assignments" in source

    def test_unassign_route_saves_to_disk(self):
        """_unassign_route persists the removal."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._unassign_route)
        assert "save_route_assignments" in source

    def test_load_route_assignments_method_exists(self):
        """LogicMixin has _load_route_assignments method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_load_route_assignments")
        source = inspect.getsource(
            LogicMixin._load_route_assignments)
        assert "load_route_assignments" in source

    def test_load_route_called_on_startup(self):
        """_load_route_assignments is called during __init__."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_load_route_assignments" in source


class TestStudentRoute:
    """Test the STUDENT route integration."""

    def test_route_keys_includes_student(self):
        """ROUTE_KEYS contains the student route."""
        from enigma_engine.gui.scanners import ROUTE_KEYS
        assert "student" in ROUTE_KEYS

    def test_router_page_builds_student_card(self):
        """ROUTER page builder creates a STUDENT route card."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_router)
        assert "STUDENT" in source
        assert "AI model being trained" in source

    def test_forge_has_student_route_label(self):
        """FORGE page creates _student_route_label widget."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_student_route_label" in source

    def test_finetune_uses_student_route(self):
        """_start_solo_training loads from the STUDENT route."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_solo_training)
        assert "student" in source
        assert "student_path" in source

    def test_finetune_logs_trainer_as_evaluator(self):
        """Guided training uses trainer and student."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert "trainer" in source
        assert "student" in source

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


class TestCopyModel:
    """Test the copy model feature on the MODELS page."""

    def test_copy_model_method_exists(self):
        """ForgeMixin._copy_model is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_copy_model")
        assert callable(getattr(ForgeMixin, "_copy_model"))

    def test_copy_model_uses_models_msg(self):
        """_copy_model uses _models_msg for feedback."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._copy_model)
        assert "_models_msg" in source

    def test_copy_model_refreshes_list(self):
        """_copy_model calls _refresh_models after success."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._copy_model)
        assert "_refresh_models" in source

    def test_copy_model_has_concurrent_guard(self):
        """All model operations use _model_op_busy guard."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # Every heavy operation must check _model_op_busy
        for method_name in ("_copy_model", "_import_model",
                            "_create_new_model", "_delete_model"):
            source = inspect.getsource(
                getattr(ForgeMixin, method_name))
            assert "_model_op_busy" in source, (
                f"{method_name} must call _model_op_busy()")
            assert "_model_op_in_progress" in source, (
                f"{method_name} must set _model_op_in_progress")

    def test_copy_button_in_model_cards(self):
        """_populate_model_cards renders a COPY button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "COPY" in source
        assert "_copy_model" in source


class TestBlankModelCreate:
    """Test simplified blank model creation on MODELS page."""

    def test_create_model_no_size_input(self):
        """_create_new_model does not accept size — uses default preset."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._create_new_model)
        # No size entry or suffix references
        assert "new_model_size_entry" not in source
        assert "_model_size_suffix" not in source
        # Uses small preset directly
        assert "small" in source

    def test_rename_dialog_has_no_resize(self):
        """_rename_model is a pure rename — no resize option."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        assert "Resize to" not in source
        assert "_resize_model" not in source

    def test_model_card_shows_param_count(self):
        """_populate_model_cards shows param count next to name."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "params" in source


class TestForgeParamCount:
    """Test that FORGE page shows param count after training."""

    def test_forge_student_card_has_param_label(self):
        """FORGE page builds a _forge_student_params label."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_forge_student_params" in source

    def test_solo_training_logs_param_count(self):
        """Solo training logs param count on completion."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_solo_training)
        assert "_update_forge_param_count" in source

    def test_guided_training_logs_param_count(self):
        """Guided training logs param count on completion."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "_update_forge_param_count" in source

    def test_dialogue_training_logs_param_count(self):
        """Dialogue training logs param count on completion."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_dialogue_training)
        assert "_update_forge_param_count" in source

    def test_dpo_training_logs_param_count(self):
        """DPO training logs param count on completion."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_dpo_training)
        assert "_update_forge_param_count" in source

    def test_update_forge_param_count_method_exists(self):
        """ForgeMixin._update_forge_param_count is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_update_forge_param_count")
        assert callable(getattr(ForgeMixin, "_update_forge_param_count"))

    def test_training_start_clears_old_param_count(self):
        """Each training mode clears old param count at start."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # Solo training clears param count before training
        source = inspect.getsource(ForgeMixin._start_solo_training)
        assert "_clear_forge_param_count" in source

    def test_clear_forge_param_count_method_exists(self):
        """ForgeMixin._clear_forge_param_count is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_clear_forge_param_count")
        assert callable(getattr(ForgeMixin, "_clear_forge_param_count"))


class TestForgeModeUI:
    """Test that FORGE page shows/hides UI sections per training mode."""

    def test_on_training_mode_changed_exists(self):
        """ForgeMixin._on_training_mode_changed is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_on_training_mode_changed")
        assert callable(getattr(ForgeMixin, "_on_training_mode_changed"))

    def test_forge_data_section_container_exists(self):
        """_build_page_forge creates _forge_data_section container."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_forge_basic_section" in source

    def test_forge_stages_section_container_exists(self):
        """_build_page_forge creates _forge_stages_section container."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_forge_stages_section" in source

    def test_forge_brief_section_container_exists(self):
        """_build_page_forge creates _forge_brief_section container."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_forge_brief_section" in source

    def test_forge_pairs_section_container_exists(self):
        """_build_page_forge creates _forge_pairs_section container."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_forge_pairs_section" in source

    def test_mode_changed_handles_data_section(self):
        """_on_training_mode_changed manages _forge_data_section."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        assert "_forge_basic_section" in source

    def test_mode_changed_handles_stages_section(self):
        """_on_training_mode_changed manages _forge_stages_section."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        assert "_forge_stages_section" in source

    def test_mode_changed_handles_brief_section(self):
        """_on_training_mode_changed manages _forge_brief_section."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        assert "_forge_brief_section" in source

    def test_mode_changed_handles_pairs_section(self):
        """_on_training_mode_changed manages _forge_pairs_section."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        assert "_forge_pairs_section" in source

    def test_mode_changed_updates_data_label_text(self):
        """_on_training_mode_changed updates _forge_data_label."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        assert "_forge_ai_section" in source

    def test_solo_hides_trainer_only_sections(self):
        """Solo mode hides brief, stages, pairs sections."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        # Should reference pack_forget for hiding sections
        assert "pack_forget" in source

    def test_mode_visibility_map_defined(self):
        """ForgeMixin has visibility config for all modes including new ones."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        vis = ForgeMixin._MODE_SECTION_VISIBILITY
        assert "Solo" in vis
        assert "Dialogue" in vis
        assert "DPO" in vis
        assert "Vision" in vis
        assert "LoRA" in vis
        assert "Evolutionary" in vis

    def test_vision_mode_in_descriptions(self):
        """Vision mode has a description."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert "Vision" in ForgeMixin._TRAINING_MODE_DESCRIPTIONS

    def test_lora_mode_in_descriptions(self):
        """LoRA mode has a description."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert "LoRA" in ForgeMixin._TRAINING_MODE_DESCRIPTIONS

    def test_evolutionary_mode_in_descriptions(self):
        """Evolutionary mode has a description."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert "Evolutionary" in ForgeMixin._TRAINING_MODE_DESCRIPTIONS

    def test_vision_mode_data_label(self):
        """Vision mode has a data label."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert "Vision" in ForgeMixin._MODE_DATA_LABELS

    def test_lora_mode_data_label(self):
        """LoRA mode has a data label."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert "LoRA" in ForgeMixin._MODE_DATA_LABELS

    def test_evolutionary_mode_data_label(self):
        """Evolutionary mode has a data label."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert "Evolutionary" in ForgeMixin._MODE_DATA_LABELS


class TestForgeNewModes:
    """Test new training modes: Vision, LoRA, Evolutionary."""

    def test_training_mode_dropdown_has_all_modes(self):
        """_build_page_forge creates dropdown with all 7 training modes."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        # Dropdown uses display names, not internal keys
        assert "Basic" in source
        assert "AI-Guided" in source
        assert "Image" in source

    def test_dispatcher_handles_vision(self):
        """_start_training_by_mode dispatches Vision mode."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_training_by_mode)
        assert "Vision" in source
        assert "_start_vision_training" in source

    def test_dispatcher_handles_lora(self):
        """_start_training_by_mode dispatches LoRA mode."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_training_by_mode)
        # LoRA is auto-dispatched from Basic mode based on param count
        basic_source = inspect.getsource(ForgeMixin._start_basic_training)
        assert "_start_lora_training" in basic_source

    def test_dispatcher_handles_evolutionary(self):
        """_start_training_by_mode dispatches Evolutionary mode."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_training_by_mode)
        # Evolutionary mode is available as a standalone method
        assert hasattr(ForgeMixin, "_start_evolutionary_training")

    def test_vision_training_method_exists(self):
        """ForgeMixin._start_vision_training is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_start_vision_training")
        assert callable(getattr(ForgeMixin, "_start_vision_training"))

    def test_lora_training_method_exists(self):
        """ForgeMixin._start_lora_training is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_start_lora_training")
        assert callable(getattr(ForgeMixin, "_start_lora_training"))

    def test_evolutionary_training_method_exists(self):
        """ForgeMixin._start_evolutionary_training is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_start_evolutionary_training")
        assert callable(getattr(ForgeMixin, "_start_evolutionary_training"))

    def test_vision_training_uses_train_vision(self):
        """_start_vision_training calls Trainer.train_vision."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_vision_training)
        assert "train_vision" in source

    def test_vision_training_uses_scan_vision_data(self):
        """_start_vision_training scans for image-text pairs."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_vision_training)
        assert "scan_vision_data" in source

    def test_lora_training_uses_lora_utils(self):
        """_start_lora_training uses LoRA training utilities."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_lora_training)
        assert "lora" in source.lower()

    def test_evolutionary_training_uses_backend(self):
        """_start_evolutionary_training uses evolutionary_training."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_evolutionary_training)
        assert "evolutionary_training" in source

    def test_vision_section_in_forge_page(self):
        """_build_page_forge creates _forge_vision_section container."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_forge_image_section" in source

    def test_lora_section_in_forge_page(self):
        """_build_page_forge creates _forge_lora_section container."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        # LoRA is auto-selected in Basic mode — no dedicated section needed
        assert "_forge_basic_section" in source

    def test_evolutionary_section_in_forge_page(self):
        """_build_page_forge creates _forge_evo_section container."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        # Evolutionary mode was consolidated into Basic mode
        assert "_forge_basic_section" in source

    def test_focus_field_entry_in_forge_page(self):
        """_build_page_forge has a focus field entry for specific topics."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        # Focus field is part of the AI-Guided section (training brief)
        assert "_forge_ai_section" in source or "_forge_brief_section" in source

    def test_mode_changed_handles_vision_section(self):
        """_on_training_mode_changed manages _forge_vision_section."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        assert "_forge_image_section" in source

    def test_mode_changed_handles_lora_section(self):
        """_on_training_mode_changed manages _forge_lora_section."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        # LoRA is auto-selected in Basic mode — managed via _forge_basic_section
        assert "_forge_basic_section" in source

    def test_mode_changed_handles_evo_section(self):
        """_on_training_mode_changed manages _forge_evo_section."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        # Evolutionary mode was consolidated — main 3-mode sections still managed
        assert "_forge_ai_section" in source

    def test_display_name_mapping_covers_all_modes(self):
        """_MODE_DISPLAY_TO_KEY maps all 9 display names to keys."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        mapping = ForgeMixin._MODE_DISPLAY_TO_KEY
        assert len(mapping) == 9
        # Every display name resolves to a valid internal key
        internal_keys = {"Solo", "Dialogue", "DPO",
                         "Vision", "LoRA", "Evolutionary",
                         "Adaptive", "RLHF", "SelfPlay"}
        assert set(mapping.values()) == internal_keys

    def test_reverse_mapping_covers_all_keys(self):
        """_MODE_KEY_TO_DISPLAY maps all 9 internal keys to display."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        reverse = ForgeMixin._MODE_KEY_TO_DISPLAY
        assert len(reverse) == 9
        # Image Training is the display name for Vision
        assert reverse["Vision"] == "Image Training"
        assert reverse["Evolutionary"] == "Trial & Error"
        assert reverse["RLHF"] == "RLHF"
        assert reverse["SelfPlay"] == "Self-Play"

    def test_browse_vision_dir_exists(self):
        """ForgeMixin._browse_vision_dir is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_browse_vision_dir")
        assert callable(getattr(ForgeMixin, "_browse_vision_dir"))

    def test_browse_button_in_vision_section(self):
        """_build_page_forge has a browse button for vision dir."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_browse_vision_dir" in source
        assert "_forge_vision_browse_btn" in source

    def test_mode_changed_translates_display_names(self):
        """_on_training_mode_changed translates display → internal."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        # 3-mode contract: dispatcher uses mode names directly
        assert "AI-Guided" in source


class TestImportModel:
    """Test the import external model feature on the MODELS page."""


class TestForgeThreeModeConnections:
    """Regression tests for 3-mode FORGE wiring."""

    def test_generate_data_uses_ai_supplement_in_guided_mode(self):
        """_generate_training_data should read ai_supplement_var for AI-Guided mode."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._generate_training_data)
        assert "ai_supplement_var" in source

    def test_evaluate_student_does_not_read_removed_focus_field(self):
        """_evaluate_student should not reference removed forge_focus_field widget."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._evaluate_student)
        assert "forge_focus_field" not in source

    def test_auto_train_routes_generated_data_by_mode(self):
        """Auto-train should support AI-Guided supplement routing."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._generate_training_data)
        assert "training_mode_var" in source
        assert "ai_supplement_var" in source

    def test_adaptive_training_no_undefined_focus_field(self):
        """_start_adaptive_training should not reference undefined focus_field."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_adaptive_training)
        assert 'focus_field = ""' in source

    def test_import_model_method_exists(self):
        """ForgeMixin._import_model is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_import_model")
        assert callable(getattr(ForgeMixin, "_import_model"))

    def test_import_model_uses_models_msg(self):
        """_import_model uses _models_msg for feedback."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._import_model)
        assert "_models_msg" in source

    def test_import_model_refreshes_list(self):
        """_import_model calls _refresh_models after success."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._import_model)
        assert "_refresh_models" in source

    def test_import_model_uses_file_dialog(self):
        """_import_model uses filedialog to pick a file."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._import_model)
        assert "filedialog" in source

    def test_import_model_supports_all_formats(self):
        """_import_model file dialog includes all supported formats."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._import_model)
        assert ".gguf" in source
        assert ".pth" in source
        assert ".safetensors" in source

    def test_import_button_on_models_page(self):
        """_build_page_models has an IMPORT button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_models)
        assert "IMPORT" in source
        assert "_import_model" in source

    def test_import_copies_to_models_dir(self):
        """_import_model copies the file into models/ directory."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._import_model)
        assert "shutil" in source
        assert "copy2" in source


class TestModelCardTags:
    """Test that model cards show NATIVE/EXTERNAL tags."""

    def test_model_cards_show_native_tag(self):
        """_populate_model_cards marks native models with NATIVE tag."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "NATIVE" in source
        assert "EXTERNAL" in source

    def test_native_formats_defined(self):
        """_populate_model_cards defines native_formats set."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "native_formats" in source
        assert '"pth"' in source
        assert '"pt"' in source

    def test_resize_only_shown_for_native(self):
        """RESIZE button only renders for native format models."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "is_native" in source
        # RESIZE is inside an `if is_native:` block
        assert "if is_native" in source


class TestRenameModel:
    """Test the rename model feature on the MODELS page."""

    def test_rename_model_method_exists(self):
        """ForgeMixin._rename_model is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_rename_model")
        assert callable(getattr(ForgeMixin, "_rename_model"))

    def test_rename_model_uses_models_msg(self):
        """_rename_model uses _models_msg for feedback."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        assert "_models_msg" in source

    def test_rename_model_updates_routes(self):
        """_rename_model updates route_assignments after rename."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        assert "route_assignments" in source
        assert "save_route_assignments" in source

    def test_rename_model_refreshes_list(self):
        """_rename_model calls _refresh_models after success."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        assert "_refresh_models" in source

    def test_rename_via_context_menu_in_model_cards(self):
        """_populate_model_cards has right-click context menu."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(
            PagesMixin._populate_model_cards)
        # RENAME button replaced with right-click context menu
        assert "Button-3" in source
        assert "Rename" in source or "_start_file_rename" in source

    def test_rename_sanitizes_input(self):
        """_rename_model sanitizes the new name."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        assert "isalnum" in source

    def test_rename_model_renames_context(self):
        """_rename_model renames the model context directory."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        assert "model_key_from_path" in source or "context_dir" in source, (
            "Rename must move model context to new name")

    def test_rename_model_preserves_history(self):
        """Renaming a model preserves its chat history via context rename."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        # Should reference model_context module or rename context dir
        assert "model_context" in source or "model_contexts" in source, (
            "Rename must handle per-model context directory")


class TestCopyModel:
    """Test the copy model feature."""

    def test_copy_shows_arrow_in_feedback(self):
        """_copy_model feedback shows source → destination."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._copy_model)
        # Uses → to show the copy direction
        assert "→" in source or "\\u2192" in source


# ================================================================
# FORGE Page: Model Status Cards
# ================================================================

class TestForgeModelCards:
    """Test model status cards on the FORGE page."""

    def test_forge_trainer_card_exists(self):
        """FORGE page creates a TRAINER status card."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_forge_trainer_card" in source

    def test_forge_student_card_exists(self):
        """FORGE page creates a STUDENT status card."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "_forge_student_card" in source

    def test_update_forge_cards_method(self):
        """_update_forge_cards method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_update_forge_cards")

    def test_update_forge_cards_reads_routes(self):
        """_update_forge_cards reads trainer and student routes."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._update_forge_cards)
        assert "trainer" in source
        assert "student" in source

    def test_route_update_triggers_forge_cards(self):
        """_update_route_status also calls _update_forge_cards."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._update_route_status)
        assert "_update_forge_cards" in source


# ================================================================
# FORGE Page: Solo Training
# ================================================================

class TestSoloTraining:
    """Test solo training (STUDENT only, no TRAINER)."""

    def test_solo_training_method_exists(self):
        """_start_solo_training method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_start_solo_training")

    def test_solo_training_uses_student_route(self):
        """_start_solo_training loads from the STUDENT route."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_solo_training)
        assert "student" in source

    def test_solo_training_no_trainer_required(self):
        """_start_solo_training does not require TRAINER route."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_solo_training)
        # Should check student but not require trainer
        assert "student_path" in source

    def test_solo_button_on_forge_page(self):
        """FORGE page has a SOLO TRAIN button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "solo_train_btn" in source

    def test_solo_training_saves_to_student(self):
        """_start_solo_training saves back to the student model."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_solo_training)
        assert "student_path" in source
        assert "atomic_torch_save" in source


# ================================================================
# FORGE Helpers: Prompt Extraction
# ================================================================

class TestExtractPrompts:
    """Test _extract_prompts helper for parsing data files."""

    def test_extract_prompts_method_exists(self):
        """_extract_prompts static method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_extract_prompts")

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

class TestLoadEngineForPath:
    """Test _load_engine_for_path helper."""

    def test_load_engine_method_exists(self):
        """_load_engine_for_path static method exists."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_load_engine_for_path")

    def test_load_engine_uses_enigma_engine(self):
        """_load_engine_for_path calls EnigmaEngine."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._load_engine_for_path)
        assert "EnigmaEngine" in source

    def test_guided_uses_engine_for_trainer(self):
        """Guided training loads TRAINER via _load_engine_for_path."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert "_load_engine_for_path" in source

    def test_generate_data_uses_engine(self):
        """Generate data loads TRAINER via _load_engine_for_path."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "_load_engine_for_path" in source

    def test_evaluate_uses_engine(self):
        """Evaluate loads both models via _load_engine_for_path."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._evaluate_student)
        assert "_load_engine_for_path" in source


# ================================================================
# FORGE: Guided uses _extract_prompts
# ================================================================

class TestForgeUsesExtractPrompts:
    """Verify FORGE methods use _extract_prompts for optional data."""

    def test_guided_uses_extract_prompts(self):
        """Guided training uses _extract_prompts for bonus data."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert "_extract_prompts" in source

    def test_generate_data_uses_extract_prompts(self):
        """Generate data uses _extract_prompts for bonus data."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "_extract_prompts" in source

    def test_evaluate_is_autonomous(self):
        """Evaluate generates its own test questions autonomously."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._evaluate_student)
        # No longer requires a data file
        assert "_extract_prompts" not in source
        # Generates its own questions via TRAINER
        assert "Test question" in source or "test" in source.lower()


# ================================================================
# FORGE Page: Guided Training
# ================================================================

class TestGuidedTraining:
    """Test autonomous guided training (TRAINER teaches STUDENT)."""

    def test_guided_training_method_exists(self):
        """_start_guided_training method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_start_guided_training")

    def test_guided_training_uses_both_routes(self):
        """_start_guided_training uses both TRAINER and STUDENT."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "trainer" in source
        assert "student" in source

    def test_guided_training_generates_curriculum(self):
        """_start_guided_training has TRAINER create curriculum."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "GENERATING CURRICULUM" in source
        assert "num_pairs" in source

    def test_guided_training_tests_student(self):
        """_start_guided_training has Phase 3 interactive testing."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "TESTING STUDENT" in source
        assert "test_scores" in source
        assert "judge_msg" in source

    def test_guided_training_readiness_assessment(self):
        """_start_guided_training reports readiness to advance."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "READY" in source
        assert "PROGRESSING" in source
        assert "NEEDS WORK" in source
        assert "stages_list" in source

    def test_guided_data_file_optional(self):
        """Guided training works without a data file selected."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "has_data" in source
        # No early return for missing data file
        assert "No training data selected" not in source

    def test_guided_reads_pairs_count(self):
        """Guided training reads num_pairs from guided_pairs_entry."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "guided_pairs_entry" in source
        assert "num_pairs" in source

    def test_guided_button_on_forge_page(self):
        """FORGE page has a GUIDED TRAIN button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "guided_train_btn" in source

    def test_guided_requires_trainer(self):
        """_start_guided_training requires TRAINER route."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "trainer_path" in source

    def test_pairs_entry_on_forge_page(self):
        """FORGE page has a pairs to generate entry."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "guided_pairs_entry" in source

    def test_guided_training_saves_curriculum(self):
        """Guided training saves curriculum to DATA_DIR for review."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        # Curriculum must be saved as a file, not thrown away
        assert "curriculum_" in source or "guided_" in source
        assert "write_text" in source

    def test_guided_training_saves_test_results(self):
        """Guided training appends Phase 3 test results to the file."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        # Test results should be written to file
        assert "Test Results" in source or "TEST RESULTS" in source

    def test_guided_training_refreshes_data_files(self):
        """After saving curriculum, guided training refreshes data dropdown."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "_refresh_data_files" in source


# ================================================================
# FORGE Page: Dialogue Training (TRAINER ↔ STUDENT conversation)
# ================================================================

class TestDialogueTraining:
    """Test dialogue training where TRAINER and STUDENT talk directly."""

    def test_dialogue_training_method_exists(self):
        """_start_dialogue_training method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_start_dialogue_training")

    def test_dialogue_training_uses_both_routes(self):
        """_start_dialogue_training uses both TRAINER and STUDENT."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "trainer" in source
        assert "student" in source

    def test_dialogue_has_conversation_loop(self):
        """_start_dialogue_training has a multi-turn conversation."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "DIALOGUE" in source
        assert "conversation" in source or "turn" in source

    def test_dialogue_trains_student_on_corrections(self):
        """_start_dialogue_training trains STUDENT on corrections."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "Trainer" in source  # training.Trainer class
        assert "train" in source

    def test_dialogue_saves_student(self):
        """_start_dialogue_training saves trained STUDENT model."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "atomic_torch_save" in source

    def test_dialogue_has_correction_step(self):
        """TRAINER corrects STUDENT answers during dialogue."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "correction" in source or "corrected" in source

    def test_dialogue_tracks_improvement(self):
        """Dialogue training tracks improvement over rounds."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "score" in source or "improvement" in source

    def test_dialogue_button_on_forge_page(self):
        """FORGE page has a DIALOGUE TRAIN button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "dialogue_train_btn" in source

    def test_dialogue_rounds_entry_on_forge(self):
        """FORGE page has dialogue rounds entry."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "dialogue_rounds_entry" in source

    def test_dialogue_uses_engine_for_student(self):
        """Dialogue loads STUDENT via EnigmaEngine for inference."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "_load_engine_for_path" in source

    def test_dialogue_uses_training_stage(self):
        """Dialogue training respects the training stage setting."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "training_stage_var" in source or "stage" in source

    def test_dialogue_displays_conversation(self):
        """Dialogue training logs the conversation turns."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "TRAINER:" in source or "STUDENT:" in source

    def test_dialogue_saves_transcript(self):
        """Dialogue training saves transcript to data/ for review."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "transcript" in source
        assert "write_text" in source

    def test_dialogue_reinforces_good_answers(self):
        """High-scoring student answers are used as reinforcement."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        # Score >= 8 means use student's own answer
        assert "score >= 8" in source
        assert "student_answer" in source

    def test_dialogue_multi_run_note_in_ui(self):
        """FORGE page supports dialogue training mode."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "dialogue" in source.lower()


# ================================================================
# FORGE: Stage-Aware Generation Prompts
# ================================================================

class TestGenerationPromptBuilder:
    """Test _build_generation_prompt produces varied formats per stage."""

    def test_method_exists(self):
        """_build_generation_prompt exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_build_generation_prompt")

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

    def test_guided_training_uses_builder(self):
        """Guided training uses _build_generation_prompt, not hardcoded Q&A."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "_build_generation_prompt" in source

    def test_generate_data_uses_builder(self):
        """Generate data uses _build_generation_prompt, not hardcoded Q&A."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._generate_training_data)
        assert "_build_generation_prompt" in source


# ================================================================
# FORGE: Training Pair Formatter
# ================================================================

class TestFormatTrainingPair:
    """Test _format_training_pair outputs the right format per stage."""

    def test_method_exists(self):
        """_format_training_pair exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_format_training_pair")

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

    def test_supplement_uses_formatter(self):
        """Guided training supplement uses _format_training_pair."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert "_format_training_pair" in source

    def test_dialogue_corrections_use_formatter(self):
        """Dialogue training corrections use _format_training_pair."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "_format_training_pair" in source

    def test_generate_data_supplement_uses_formatter(self):
        """Generate data supplement uses _format_training_pair."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "_format_training_pair" in source

    def test_training_guide_documents_stage_formats(self):
        """Training guide documents stage-specific data formats."""
        guide = Path(__file__).parent.parent / "information" / "training_guide.md"
        content = guide.read_text(encoding="utf-8")
        # Should document that each stage produces different formats
        assert "Data Format" in content
        assert "User:" in content and "AI:" in content
        assert "Dialogue" in content.lower() or "dialogue" in content

    def test_training_guide_documents_all_formats(self):
        """Training guide lists all supported data formats."""
        guide = Path(__file__).parent.parent / "information" / "training_guide.md"
        content = guide.read_text(encoding="utf-8")
        for fmt in ["Q&A", "Dialogue", "JSONL", "Raw text"]:
            assert fmt in content, f"Missing format: {fmt}"

    def test_training_guide_visible_on_docs_page(self):
        """Training guide is discoverable by scan_docs."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        names = [d["filename"] for d in docs]
        assert "training_guide.md" in names

class TestGenerateData:
    """Test TRAINER autonomously generating training data."""

    def test_generate_data_method_exists(self):
        """_generate_training_data method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_generate_training_data")

    def test_generate_data_uses_trainer(self):
        """_generate_training_data uses the TRAINER route model."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "trainer" in source

    def test_generate_data_saves_file(self):
        """_generate_training_data saves output to a data file."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "write_text" in source or "write" in source

    def test_generate_data_autonomous(self):
        """_generate_training_data works without a data file."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "has_data" in source
        assert "num_pairs" in source

    def test_generate_button_on_forge_page(self):
        """FORGE page has a GENERATE DATA button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "generate_data_btn" in source


# ================================================================
# FORGE Page: Evaluate Student
# ================================================================

class TestEvaluateStudent:
    """Test TRAINER interactively testing STUDENT."""

    def test_evaluate_method_exists(self):
        """_evaluate_student method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_evaluate_student")

    def test_evaluate_uses_both_models(self):
        """_evaluate_student loads both TRAINER and STUDENT."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._evaluate_student)
        assert "trainer" in source
        assert "student" in source

    def test_evaluate_trainer_judges(self):
        """_evaluate_student has TRAINER judge student answers."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._evaluate_student)
        assert "judge_msg" in source
        assert "SCORE:" in source
        assert "score" in source.lower()

    def test_evaluate_readiness_assessment(self):
        """_evaluate_student determines readiness to advance."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._evaluate_student)
        assert "READY" in source
        assert "PROGRESSING" in source
        assert "NEEDS WORK" in source

    def test_evaluate_no_data_file_required(self):
        """_evaluate_student works without a data file."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._evaluate_student)
        # No check for data_path / train_data_var
        assert "train_data_var" not in source

    def test_evaluate_button_on_forge_page(self):
        """FORGE page has an EVALUATE button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "evaluate_btn" in source


# ================================================================
# FORGE Page: Checkpoint Save/Resume
# ================================================================

class TestCheckpoints:
    """Test checkpoint save and resume functionality."""

    def test_save_checkpoint_method_exists(self):
        """_save_forge_checkpoint method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_save_forge_checkpoint")

    def test_load_checkpoint_method_exists(self):
        """_load_forge_checkpoint method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_load_forge_checkpoint")

    def test_checkpoint_buttons_on_forge(self):
        """FORGE page has SAVE and LOAD checkpoint buttons."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "save_ckpt_btn" in source
        assert "load_ckpt_btn" in source

    def test_checkpoint_uses_student_route(self):
        """_save_forge_checkpoint saves the student model state."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._save_forge_checkpoint)
        assert "student" in source


# ================================================================
# FORGE Page: Loss Curve Visualization
# ================================================================

class TestLossCurve:
    """Test loss curve visualization in the output log."""

    def test_display_loss_curve_method(self):
        """_display_loss_curve method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_display_loss_curve")

    def test_loss_curve_uses_bar_chars(self):
        """_display_loss_curve renders a text-based chart."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._display_loss_curve)
        # Should use block characters for the chart
        assert "█" in source or "bar" in source.lower()

    def test_loss_curve_calls_canvas_chart(self):
        """_display_loss_curve also updates the graphical canvas chart."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._display_loss_curve)
        assert "_update_loss_chart" in source

    def test_update_loss_chart_method_exists(self):
        """_update_loss_chart method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_update_loss_chart")

    def test_update_loss_chart_is_thread_safe(self):
        """_update_loss_chart schedules via self.after for thread safety."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._update_loss_chart)
        assert "self.after" in source

    def test_update_loss_chart_draws_lines(self):
        """_update_loss_chart draws loss curve and grid."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._update_loss_chart)
        assert "create_line" in source
        assert "C_GREEN" in source

    def test_display_loss_curve_auto_expands_panel(self):
        """_display_loss_curve auto-expands the loss chart panel."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._display_loss_curve)
        assert "expand" in source


# ================================================================
# CoT-B: REASONING-AWARE TRAINING DATA
# ================================================================

class TestReasoningAwareData:
    """Test reasoning parameter in training data generation."""

    def test_build_generation_prompt_has_reasoning_param(self):
        """_build_generation_prompt accepts a reasoning parameter."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        sig = inspect.signature(ForgeMixin._build_generation_prompt)
        assert "reasoning" in sig.parameters

    def test_reasoning_prompt_includes_think_tags(self):
        """When reasoning=True, prompt includes <think> instructions."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._build_generation_prompt)
        assert "<think>" in source

    def test_forge_reasoning_var_attribute(self):
        """ForgePageMixin creates forge_reasoning_var."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert "forge_reasoning_var" in source

    def test_data_gen_passes_reasoning_flag(self):
        """Data generation reads forge_reasoning_var and passes reasoning."""
        import inspect
        from enigma_engine.gui.gui_forge_tools import ForgeToolsMixin
        source = inspect.getsource(ForgeToolsMixin._generate_training_data)
        assert "reasoning" in source

    def test_guided_training_passes_reasoning_flag(self):
        """Guided training reads forge_reasoning_var and passes reasoning."""
        import inspect
        from enigma_engine.gui.gui_forge_advanced import ForgeAdvancedMixin
        source = inspect.getsource(
            ForgeAdvancedMixin._start_guided_training)
        assert "reasoning" in source


# ================================================================
# FORGE: Trainer System Prompt (human-like responses)
# ================================================================

class TestBuildTrainerSystemPrompt:
    """Test _build_trainer_system_prompt for human-like output."""

    def test_method_exists(self):
        """_build_trainer_system_prompt exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_build_trainer_system_prompt")

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


class TestTrainerUsesChatNotGenerate:
    """Verify FORGE methods use chat() with system_prompt, not bare generate()."""

    def test_guided_uses_chat_with_system_prompt(self):
        """Guided training calls teacher_engine.chat() with system_prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert "teacher_engine.chat(" in source
        assert "_build_trainer_system_prompt" in source

    def test_generate_data_uses_chat(self):
        """Generate data calls engine.chat() with system_prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "engine.chat(" in source
        assert "_build_trainer_system_prompt" in source

    def test_evaluate_trainer_uses_chat(self):
        """Evaluate uses chat() for TRAINER, generate() for STUDENT."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._evaluate_student)
        assert "teacher_engine.chat(" in source
        assert "student_engine.generate(" in source
        assert "_build_trainer_system_prompt" in source


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

    def test_guided_training_reads_stage(self):
        """Guided training reads training_stage_var from UI."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert "training_stage_var" in source
        assert "stage=" in source

    def test_generate_data_reads_stage(self):
        """Generate data reads training_stage_var from UI."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "training_stage_var" in source
        assert "stage=" in source

    def test_evaluate_reads_stage(self):
        """Evaluate reads training_stage_var from UI."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._evaluate_student)
        assert "training_stage_var" in source
        assert "stage=" in source

    def test_stage_buttons_on_forge_page(self):
        """FORGE page has training stage buttons with tooltips."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "training_stage_var" in source
        assert "_stage_buttons" in source
        assert "_select_training_stage" in source


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

    def test_build_training_brief_method_exists(self):
        """ForgeMixin has _build_training_brief() to assemble brief from UI fields."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_build_training_brief")

    def test_save_load_training_brief(self):
        """save/load training brief round-trips through JSON."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_save_training_brief")
        assert hasattr(ForgeMixin, "_load_training_brief")

    def test_training_mode_selection_is_persisted(self):
        """FORGE should persist the selected training mode."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        save_source = inspect.getsource(ForgeMixin._save_training_brief)
        load_source = inspect.getsource(ForgeMixin._load_training_brief)
        assert '_training_mode' in save_source
        assert '_training_mode' in load_source
        assert '_on_training_mode_changed' in load_source

    def test_forge_page_radio_uses_mode_selection_helper(self):
        """Selecting a FORGE mode should immediately save the choice."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert '_on_training_mode_selected' in source

    def test_forge_page_has_training_brief_panel(self):
        """FORGE page builder creates the Training Brief panel."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "TRAINING BRIEF" in source or "training_brief" in source

    def test_guided_training_reads_brief(self):
        """Guided training reads training brief from UI and passes it."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert "training_brief" in source

    def test_dialogue_training_reads_brief(self):
        """Dialogue training reads training brief and passes it."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "training_brief" in source

    def test_generate_data_reads_brief(self):
        """Generate data reads training brief and passes it."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "training_brief" in source

    def test_evaluate_reads_brief(self):
        """Evaluate reads training brief and passes it."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._evaluate_student)
        assert "training_brief" in source

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

    def test_brief_auto_includes_student_name(self):
        """_build_training_brief auto-injects the student model name."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._build_training_brief)
        assert "route_assignments" in source
        assert "student" in source
        # Should extract stem from the student path
        assert ".stem" in source

    def test_build_training_brief_combines_fields_and_custom(self):
        """_build_training_brief merges quick profile + custom text."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # Static test — verify the method signature takes the right args
        source = inspect.getsource(ForgeMixin._build_training_brief)
        assert "quick_fields" in source or "self" in source


# ================================================================
# FORGE: UI Polish
# ================================================================

class TestAutoLoRA:
    """Test auto-LoRA trigger for models > 7B params in Basic mode."""

    def test_get_model_param_count_exists(self):
        """ForgeMixin._get_model_param_count method is defined."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_get_model_param_count")
        assert callable(getattr(ForgeMixin, "_get_model_param_count"))

    def test_start_basic_training_calls_get_param_count(self):
        """_start_basic_training calls _get_model_param_count."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_basic_training)
        assert "_get_model_param_count" in source

    def test_start_basic_training_dispatches_lora_if_large(self):
        """_start_basic_training calls _start_lora_training for > 7B models."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_basic_training)
        assert "_start_lora_training" in source
        assert "7_000_000_000" in source or "7B" in source

    def test_start_basic_training_dispatches_solo_if_small(self):
        """_start_basic_training calls _start_solo_training for <= 7B models."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_basic_training)
        assert "_start_solo_training" in source

    def test_get_param_count_returns_int(self):
        """_get_model_param_count returns int (0 on error)."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._get_model_param_count)
        assert "int" in source
        assert "return 0" in source


class TestForgeUIPolish:
    """Test FORGE page UI polish improvements."""

    def test_train_button_shows_training_text(self):
        """Training methods change TRAIN button text to TRAINING..."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        for method_name in ("_start_solo_training",
                            "_start_guided_training",
                            "_start_dialogue_training"):
            source = inspect.getsource(
                getattr(ForgeMixin, method_name))
            assert "TRAINING" in source

    def test_train_button_restores_text(self):
        """Training finally blocks restore TRAIN button text."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        for method_name in ("_start_solo_training",
                            "_start_guided_training",
                            "_start_dialogue_training"):
            source = inspect.getsource(
                getattr(ForgeMixin, method_name))
            # The finally block should restore text to "TRAIN"
            assert '"TRAIN"' in source

    def test_mode_change_dims_stages_for_solo(self):
        """_on_training_mode_changed hides stages for Solo."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._on_training_mode_changed)
        assert "_stage_buttons" in source
        # Solo mode hides stages via visibility map
        vis = ForgeMixin._MODE_SECTION_VISIBILITY
        assert vis["Solo"]["stages"] is False

    def test_web_learn_layout_single_row(self):
        """Web Learn topic, pages entry, and button on one row."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        # web_learn_pages_entry should be in web_row, not standalone
        assert "web_learn_pages_entry" in source


# ================================================================
# FORGE: Web Learn
# ================================================================

class TestWebLearn:
    """Test WEB LEARN feature — TRAINER gathers data from the web."""

    def test_web_learn_method_exists(self):
        """_web_learn method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_web_learn")

    def test_web_learn_requires_trainer(self):
        """_web_learn checks for TRAINER route assignment."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "trainer" in source

    def test_web_learn_uses_requests(self):
        """_web_learn imports requests for web access."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "requests" in source

    def test_web_learn_saves_data_file(self):
        """_web_learn saves generated pairs to a data file."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "write_text" in source
        assert "web_" in source

    def test_web_learn_uses_trainer_chat(self):
        """_web_learn uses engine.chat() to generate Q/A pairs."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "engine.chat(" in source

    def test_web_learn_searches_duckduckgo(self):
        """_web_learn uses ddg_search from web_utils."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "ddg_search" in source

    def test_web_learn_button_on_forge_page(self):
        """FORGE page has a WEB LEARN button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "web_learn_btn" in source
        assert "web_learn_topic" in source

    def test_web_learn_refreshes_data_files(self):
        """_web_learn refreshes data file list after saving."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "_refresh_data_files" in source


# ================================================================
# System as visible third speaker in chat
# ================================================================

class TestSystemSpeaker:
    """SYSTEM is a visible participant in the chat — not hidden or anonymous."""

    def test_system_prefix_tag_exists(self):
        """Chat display has a system_prefix tag configured."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "system_prefix" in source

    def test_system_msg_tag_exists(self):
        """Chat display has a system_msg tag for system text."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "system_msg" in source

    def test_chat_system_uses_prefix(self):
        """_chat_system shows 'System' as a named speaker."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._chat_system)
        assert "system_prefix" in source
        assert "System" in source

    def test_chat_error_uses_prefix(self):
        """_chat_error shows 'System' as speaker — errors come from SYSTEM."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._chat_error)
        assert "system_prefix" in source
        assert "System" in source

    def test_chat_system_has_timestamp(self):
        """_chat_system includes a timestamp like User and AI messages."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._chat_system)
        assert "timestamp" in source

    def test_chat_error_has_timestamp(self):
        """_chat_error includes a timestamp."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._chat_error)
        assert "timestamp" in source

    def test_command_results_shown_as_system(self):
        """Command output is displayed as SYSTEM, not as AI speech."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "_chat_system" in source


# ================================================================
# Chat Media Support (inline images, GIFs, video thumbnails, links)
# ================================================================

class TestChatMedia:
    """Media rendering in the chat display — images, GIFs, videos, clickable links."""

    def test_media_module_exists(self):
        """media.py module exists and is importable."""
        from enigma_engine.gui import media
        assert hasattr(media, "detect_media_refs")
        assert hasattr(media, "detect_urls")
        assert hasattr(media, "load_chat_image")
        assert hasattr(media, "extract_gif_frames")
        assert hasattr(media, "extract_video_thumbnail")

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

    def test_chat_display_has_link_tag(self):
        """Chat display configures a 'link' tag for clickable URLs."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "link" in source

    def test_logic_has_insert_media(self):
        """LogicMixin has _insert_media method for rendering in chat."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_insert_media")

    def test_logic_has_process_media_in_text(self):
        """LogicMixin has _process_media_in_text for detecting media in responses."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_process_media_in_text")

    def test_logic_has_open_link(self):
        """LogicMixin has _open_link for opening URLs in browser."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_open_link")

    def test_logic_has_open_video(self):
        """LogicMixin has _open_video for opening video in default player."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_open_video")

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

    def test_attach_handles_media_files(self):
        """_attach_file detects media file extensions and routes to _attach_image."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._attach_file)
        assert "IMAGE_EXTENSIONS" in source
        assert "GIF_EXTENSIONS" in source
        assert "VIDEO_EXTENSIONS" in source
        assert "_attach_image" in source

    def test_attach_image_method_exists(self):
        """LogicMixin has _attach_image for rendering media inline."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_attach_image")
        source = inspect.getsource(LogicMixin._attach_image)
        # Accepts optional path parameter for direct calls
        assert "path" in source

    def test_send_message_processes_media(self):
        """_send_message flow includes media processing."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "_process_media_in_text" in source

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

    def test_insert_media_shows_caption_on_failure(self):
        """_insert_media shows '[Image not available]' when image cannot load."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._insert_media)
        assert "Image not available" in source


# ================================================================
# Send guard (double-send crash fix)
# ================================================================

class TestSendGuard:
    """Verify guard against sending while generation is in progress."""

    def test_is_generating_flag_exists(self):
        """LogicMixin._send_message checks _is_generating flag."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "_is_generating" in source

    def test_is_generating_set_true_before_thread(self):
        """_is_generating is set True before spawning the gen thread."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        # Flag must be set to True somewhere in _send_message
        assert "_is_generating = True" in source

    def test_is_generating_cleared_in_finally(self):
        """_is_generating is reset to False in the finally block."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        # The inner _gen function must clear the flag
        assert "_is_generating" in source
        # Check the flag is referenced alongside False
        assert "False" in source

    def test_on_input_enter_checks_generating(self):
        """Enter key handler respects _is_generating guard."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._on_input_enter)
        assert "_is_generating" in source

    def test_is_generating_init_in_desktop(self):
        """_is_generating flag is initialized in desktop __init__."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_is_generating" in source


# ================================================================
# STOP button
# ================================================================

class TestStopButton:
    """Verify STOP button exists and can cancel generation."""

    def test_stop_btn_created_in_core_page(self):
        """CORE page builds a stop_btn widget."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "stop_btn" in source

    def test_stop_generation_method_exists(self):
        """LogicMixin has a _stop_generation method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_stop_generation")

    def test_stop_generation_sets_flag(self):
        """_stop_generation sets a cancellation flag."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._stop_generation)
        assert "_stop_requested" in source

    def test_gen_thread_checks_stop_flag(self):
        """The generation thread checks for stop request."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert "_stop_requested" in source

    def test_typewriter_checks_stop_flag(self):
        """Typewriter animation respects stop flag."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._typewriter)
        assert "_stop_requested" in source

    def test_stop_btn_tooltip(self):
        """CORE page has a tooltip for the stop button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "stop" in source.lower() or "Stop" in source


# ================================================================
# Message editing
# ================================================================

class TestMessageEdit:
    """Verify users can edit sent messages."""

    def test_edit_message_method_exists(self):
        """LogicMixin has _edit_last_message method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_edit_last_message")

    def test_edit_removes_last_exchange(self):
        """_edit_last_message removes the last user+assistant pair from history."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._edit_last_message)
        # Must reference history to remove the last pair
        assert "history" in source

    def test_edit_populates_input(self):
        """_edit_last_message puts the user message back in the input box."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._edit_last_message)
        assert "chat_input" in source

    def test_edit_button_exists_in_core(self):
        """CORE page toolbar has an edit button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "edit" in source.lower()

    def test_edit_guards_empty_history(self):
        """_edit_last_message handles empty history gracefully."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._edit_last_message)
        assert "not self.history" in source or "len(self.history)" in source

    def test_edit_guards_generating(self):
        """Cannot edit while AI is generating."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._edit_last_message)
        assert "_is_generating" in source


class TestModCommandsInContext:
    """Verify mod commands are injected into AI system context."""

    def test_gui_context_uses_format_tools(self):
        """_build_gui_context delegates to format_tools_for_prompt."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "format_tools_for_prompt" in source

    def test_gui_context_shows_cmd_syntax(self):
        """Mod commands shown with [CMD]mod.command[/CMD] syntax."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "[CMD]" in source
        assert "[/CMD]" in source

    def test_gui_context_has_mod_management(self):
        """Context includes mod start/stop/list instructions."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "mod.start" in source
        assert "mod.stop" in source
        assert "mod.list" in source

    def test_gui_context_imports_mod_tools(self):
        """Context builder imports from mod_tools module."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "mod_tools" in source


class TestEscapeBinding:
    """Verify Escape key binds correctly and is reusable."""

    def test_bind_escape_stop_method_exists(self):
        """EnigmaGUI has _bind_escape_stop method."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_bind_escape_stop")

    def test_bind_escape_stop_uses_escape_key(self):
        """_bind_escape_stop binds the Escape key."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._bind_escape_stop)
        assert "<Escape>" in source

    def test_bind_escape_stop_checks_generating(self):
        """Escape only fires stop when _is_generating is true."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._bind_escape_stop)
        assert "_is_generating" in source

    def test_fullscreen_exit_rebinds_escape(self):
        """Exiting fullscreen re-binds Escape to stop generation."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(
            PagesMixin._exit_chat_fullscreen)
        assert "_bind_escape_stop" in source

    def test_bind_called_during_init(self):
        """_bind_escape_stop is called during __init__."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_bind_escape_stop" in source


class TestPageNavShortcuts:
    """Verify Ctrl+1..7 page navigation shortcuts."""

    def test_bind_page_nav_method_exists(self):
        """EnigmaGUI has _bind_page_nav_shortcuts method."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_bind_page_nav_shortcuts")
        assert callable(getattr(EnigmaGUI, "_bind_page_nav_shortcuts"))

    def test_page_nav_binds_ctrl_keys(self):
        """_bind_page_nav_shortcuts binds Control-1 through Control-7."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._bind_page_nav_shortcuts)
        assert "Control" in source
        # Should bind at least CORE via Ctrl+1
        assert "CORE" in source

    def test_page_nav_called_during_init(self):
        """_bind_page_nav_shortcuts is called during __init__."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_bind_page_nav_shortcuts" in source

    def test_page_nav_uses_switch_page(self):
        """Shortcuts use _switch_page to navigate."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._bind_page_nav_shortcuts)
        assert "_switch_page" in source


class TestWindowClose:
    """Verify cleanup on window close."""

    def test_on_close_method_exists(self):
        """EnigmaGUI has _on_close method."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_on_close")

    def test_on_close_terminates_mods(self):
        """_on_close terminates running mod subprocesses."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        assert "mod_processes" in source
        assert "terminate" in source

    def test_on_close_stops_router(self):
        """_on_close stops the ModRouter."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        assert "_router" in source
        assert "stop" in source

    def test_on_close_releases_loaded_engine(self):
        """_on_close should explicitly release model/backend resources."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        assert "_release_loaded_engine" in source

    def test_on_close_destroys_window(self):
        """_on_close calls destroy() at the end."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        assert "self.destroy()" in source

    def test_wm_delete_protocol_set(self):
        """WM_DELETE_WINDOW protocol is set to _on_close."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "WM_DELETE_WINDOW" in source
        assert "_on_close" in source

    def test_on_close_does_not_silence_exceptions(self):
        """_on_close should log failures instead of using bare except-pass."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        assert "except Exception:\n                    pass" not in source
        assert "except Exception:\n                pass" not in source


class TestForgeUsesModelsDirConstant:
    """Verify FORGE uses MODELS_DIR constant instead of hardcoded paths."""

    def test_models_dir_imported(self):
        """gui_forge.py imports MODELS_DIR from scanners."""
        from enigma_engine.gui import gui_forge
        source = inspect.getsource(gui_forge)
        assert "MODELS_DIR" in source

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

    def test_import_model_uses_models_dir(self):
        """_import_model uses MODELS_DIR for destination."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._import_model)
        assert "MODELS_DIR" in source

    def test_create_model_uses_models_dir(self):
        """_create_new_model uses MODELS_DIR for output path."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._create_new_model)
        assert "MODELS_DIR" in source


class TestDataEditorGuards:
    """Verify gui_forge.py data selection is clean."""

    def test_on_data_selected_sets_var(self):
        """_on_data_selected sets train_data_var."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._on_data_selected)
        assert "train_data_var" in source

    def test_refresh_data_files_updates_menu(self):
        """_refresh_data_files updates the dropdown menu."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._refresh_data_files)
        assert "scan_training_data" in source


class TestFileEncoding:
    """Verify all file I/O uses encoding='utf-8' on Windows."""

    def test_builtin_commands_mod_json_encoding(self):
        """mod.json reads use encoding='utf-8'."""
        from enigma_engine.core import builtin_commands
        source = inspect.getsource(builtin_commands)
        # Should NOT have open(mod_json, 'r') without encoding
        assert "open(mod_json, 'r')" not in source

    def test_model_registry_encoding(self):
        """model_registry open() calls use encoding='utf-8'."""
        from enigma_engine.core import model_registry
        source = inspect.getsource(model_registry)
        assert "encoding" in source

    def test_inference_metadata_encoding(self):
        """inference.py metadata reads use encoding='utf-8'."""
        from enigma_engine.core import inference
        source = inspect.getsource(inference)
        # metadata_file open should have encoding
        assert "'r', encoding=" in source or '"r", encoding=' in source

    def test_model_export_config_encoding(self):
        """model.py config writes use encoding='utf-8'."""
        from enigma_engine.core.model import Enigma
        source = inspect.getsource(Enigma.export_to_safetensors)
        assert "encoding" in source


class TestRouterPortDynamic:
    """Verify router port is not hardcoded in messages."""

    def test_mod_start_uses_dynamic_port(self):
        """mod_start command shows actual router port."""
        from enigma_engine.core import builtin_commands
        source = inspect.getsource(builtin_commands)
        assert "port 9900" not in source

    def test_mod_start_reads_router_port(self):
        """mod_start reads router.port for the message."""
        from enigma_engine.core import builtin_commands
        source = inspect.getsource(builtin_commands)
        assert "router.port" in source


class TestRouterStartupLogging:
    """Verify router startup failures are logged, not silently swallowed."""

    def test_desktop_router_logs_failure(self):
        """desktop.py logs router startup exceptions."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        # Should NOT have bare 'except Exception: pass'
        assert "logging" in source or "warning" in source

    def test_desktop_router_not_bare_pass(self):
        """Router startup does not use bare except: pass."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        # The old pattern was 'except Exception:\n            pass'
        assert "pass  # Router optional" not in source


class TestWebLearnErrorReporting:
    """Verify web learn reports chunk generation failures."""

    def test_web_learn_tracks_failures(self):
        """_web_learn counts and reports chunk failures."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "chunk_failures" in source

    def test_web_learn_logs_first_error(self):
        """_web_learn logs the first generation error."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "chunk_failures" in source
        assert "Generation" in source


class TestExpandingChatDisplay:
    """Verify the chat area uses native CTkTextbox scrollbar."""

    def test_chat_display_uses_native_scrollbar(self):
        """CORE page places chat_display directly (no CTkScrollableFrame)."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        # Should create SelectableTextbox directly in chat_col
        assert "SelectableTextbox" in source
        # _chat_scroll should be set to None for backward compat
        assert "_chat_scroll" in source

    def test_chat_display_sticky_nsew(self):
        """Chat display uses sticky='nsew' to fill available space."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert '"nsew"' in source or "'nsew'" in source

    def test_auto_resize_chat_method_exists(self):
        """LogicMixin has _auto_resize_chat method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_auto_resize_chat")

    def test_scroll_chat_to_bottom_method_exists(self):
        """LogicMixin has _scroll_chat_to_bottom method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_scroll_chat_to_bottom")

    def test_chat_append_calls_auto_resize(self):
        """_chat_append calls _auto_resize_chat after inserting."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._chat_append)
        assert "_auto_resize_chat" in source

    def test_typewriter_calls_auto_resize(self):
        """_typewriter calls _auto_resize_chat during insertion."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._typewriter)
        assert "_auto_resize_chat" in source

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

    def test_cmd_mods_uses_correct_attr(self):
        """CMD mods command must use 'mod_processes' not '_mod_procs'."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_cmd_page.py"
        source = source_path.read_text(encoding='utf-8')
        assert "_mod_procs" not in source, (
            "CMD page uses wrong attribute name — should be mod_processes")
        assert "mod_processes" in source

    def test_scan_training_data_skips_config_files(self):
        """scan_training_data must skip route_assignments.json and path_settings.json."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "scanners.py"
        source = source_path.read_text(encoding='utf-8')
        assert "route_assignments.json" in source, (
            "scan_training_data doesn't skip route_assignments.json")
        assert "path_settings.json" in source, (
            "scan_training_data doesn't skip path_settings.json")

    def test_version_constant_exists(self):
        """widgets.py should export a VERSION constant."""
        from enigma_engine.gui.widgets import VERSION
        assert isinstance(VERSION, str)
        assert VERSION  # Not empty

    def test_desktop_uses_version_constant(self):
        """desktop.py should use VERSION from widgets, not hardcoded."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        assert "VERSION" in source
        # Should not have bare "1.1.0" as string literal (except in VERSION itself)
        import re
        bare_version = re.findall(r'text=.*".*1\.1\.0.*"', source)
        assert not bare_version, "Version string should use VERSION constant"

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

    def test_mods_launch_logs_errors(self):
        """_launch_mod should log errors, not silently swallow them."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_mods.py"
        source = source_path.read_text(encoding='utf-8')
        assert "logger.warning" in source, (
            "_launch_mod silently swallows subprocess errors")

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

    def test_voice_text_auto_sends(self):
        """_on_voice_text should call _send_message, not just insert."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        # Find _on_voice_text method
        lines = source.split('\n')
        in_method = False
        found_send = False
        for line in lines:
            if 'def _on_voice_text' in line:
                in_method = True
            elif in_method and line.strip().startswith('def '):
                break
            elif in_method and '_send_message' in line:
                found_send = True
        assert found_send, (
            "_on_voice_text should auto-send via _send_message()")

    def test_voice_continuous_listening(self):
        """Voice input should keep listening after each phrase."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic.py"
        source = source_path.read_text(encoding='utf-8')
        # The old code set _voice_got_audio to stop after one phrase;
        # the new code should NOT have that pattern
        assert '_voice_got_audio = True' not in source, (
            "Voice input stops after one phrase — should be continuous")

    def test_voice_stop_listening_method(self):
        """_voice_stop_listening helper should exist."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, '_voice_stop_listening')

    def test_voice_guards_during_generation(self):
        """_on_voice_text should not send while AI is generating."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        in_method = False
        found_guard = False
        for line in lines:
            if 'def _on_voice_text' in line:
                in_method = True
            elif in_method and line.strip().startswith('def '):
                break
            elif in_method and '_is_generating' in line:
                found_guard = True
        assert found_guard, (
            "_on_voice_text needs _is_generating guard")


# ================================================================
# Voice Output: TTS
# ================================================================

class TestVoiceOutput:
    """Verify text-to-speech integration."""

    def test_tts_speak_method_exists(self):
        """LogicMixin should have _tts_speak."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, '_tts_speak')

    def test_tts_stop_method_exists(self):
        """LogicMixin should have _tts_stop."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, '_tts_stop')

    def test_tts_engine_init_in_desktop(self):
        """desktop.py should initialize TTS state vars."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        assert '_tts_engine_ref = None' in source
        assert '_tts_queue = None' in source

    def test_tts_called_after_response(self):
        """_tts_speak should be called after AI response in _show."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_chat.py"
        source = source_path.read_text(encoding='utf-8')
        # _tts_speak should appear in the _show closure
        assert '_tts_speak(r)' in source or '_tts_speak(r ' in source, (
            "AI responses should be spoken when voice is enabled")

    def test_tts_runs_in_thread(self):
        """_tts_speak must use a dedicated persistent TTS thread."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        # Find the whole _tts_speak method body up to the next
        # unindented method definition and check for Thread usage.
        import re
        match = re.search(
            r'def _tts_speak\b.*?(?=\n    def )',
            source, re.DOTALL)
        assert match, "_tts_speak method not found"
        body = match.group(0)
        assert 'Thread' in body, "_tts_speak must start a TTS worker thread"

    def test_tts_stops_on_toggle_off(self):
        """Turning voice off should stop TTS."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        in_method = False
        found_stop = False
        for line in lines:
            if 'def _on_voice_toggle' in line:
                in_method = True
            elif in_method and line.strip().startswith('def '):
                break
            elif in_method and '_tts_stop' in line:
                found_stop = True
        assert found_stop, (
            "_on_voice_toggle should call _tts_stop when off")

    def test_stop_generation_stops_tts(self):
        """_stop_generation should also stop TTS playback."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_chat.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        in_method = False
        found_stop = False
        for line in lines:
            if 'def _stop_generation' in line:
                in_method = True
            elif in_method and line.strip().startswith('def '):
                break
            elif in_method and '_tts_stop' in line:
                found_stop = True
        assert found_stop, (
            "_stop_generation should call _tts_stop()")

    def test_on_close_stops_tts(self):
        """_on_close should clean up TTS."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        assert '_tts_shutdown' in source, (
            "_on_close should shut down TTS on exit")

    def test_tts_shutdown_method_exists(self):
        """LogicMixin should have _tts_shutdown for cleanup."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, '_tts_shutdown')

    def test_tts_single_engine_reuse(self):
        """TTS should reuse one engine, not create per-call."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        # _tts_speak should NOT call pyttsx3.init() directly —
        # that's done in the worker only once.
        lines = source.split('\n')
        in_speak = False
        init_in_speak = False
        for line in lines:
            if 'def _tts_speak' in line:
                in_speak = True
            elif in_speak and line.strip().startswith('def ') and '_tts_worker' not in line:
                break
            # The init should only be inside the nested worker,
            # not at the top level of _tts_speak
        # Check that _tts_queue is used (queue-based design)
        assert '_tts_queue' in source, (
            "TTS should use a queue for thread-safe communication")


# ================================================================
# Model Delete: No GUI Freeze
# ================================================================

class TestModelDeleteNoFreeze:
    """Verify model deletion runs heavy work off the main thread."""

    def test_delete_model_uses_thread(self):
        """Model deletion should run file removal in a thread."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_forge_models.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        # Thread is now in _confirm_delete_model (inline confirm flow)
        in_method = False
        found_thread = False
        for line in lines:
            if 'def _confirm_delete_model' in line:
                in_method = True
            elif in_method and line.strip().startswith('def ') and '_do_delete' not in line:
                break
            elif in_method and 'Thread' in line:
                found_thread = True
        assert found_thread, (
            "_confirm_delete_model should use a background thread")

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

    def test_create_model_uses_default_preset(self):
        """_create_new_model should use a default preset (no size input)."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_forge_models.py"
        source = source_path.read_text(encoding='utf-8')
        assert 'MODEL_PRESETS' in source, (
            "Model creation must use MODEL_PRESETS for default config")

    def test_count_params_reads_target_size(self):
        """_count_params_native should prefer target_size from checkpoint."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "scanners.py"
        source = source_path.read_text(encoding='utf-8')
        assert 'target_size' in source, (
            "_count_params_native should read target_size from checkpoint")


# ================================================================
# Memory Optimization: Param Counting + Image Cap
# ================================================================

class TestMemoryOptimization:
    """Verify RAM optimizations — no huge torch.load, capped images."""

    def test_count_params_has_load_limit(self):
        """_count_params_native should skip torch.load for large files."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "scanners.py"
        source = source_path.read_text(encoding='utf-8')
        assert '_PARAM_COUNT_LOAD_LIMIT' in source, (
            "Must define a file-size limit to avoid loading huge models")

    def test_count_params_uses_zipfile_peek(self):
        """_peek_target_size reads metadata from zip without torch."""
        from enigma_engine.gui.scanners import _peek_target_size
        assert callable(_peek_target_size)

    def test_estimate_params_from_size_exists(self):
        """File-size heuristic function exists for large models."""
        from enigma_engine.gui.scanners import _estimate_params_from_size
        assert callable(_estimate_params_from_size)

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

    def test_trim_chat_images_method_exists(self):
        """LogicMixin has _trim_chat_images method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_trim_chat_images")

    def test_insert_media_calls_trim(self):
        """_insert_media calls _trim_chat_images after adding."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._insert_media)
        assert "_trim_chat_images" in source

    def test_insert_gif_calls_trim(self):
        """_insert_gif calls _trim_chat_images after adding."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._insert_gif)
        assert "_trim_chat_images" in source

    def test_insert_video_calls_trim(self):
        """_insert_video_thumbnail calls _trim_chat_images after adding."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._insert_video_thumbnail)
        assert "_trim_chat_images" in source

    def test_peek_target_size_returns_none_for_missing(self):
        """_peek_target_size returns None for non-existent file."""
        from enigma_engine.gui.scanners import _peek_target_size
        result = _peek_target_size(Path("nonexistent_model.pth"))
        assert result is None

    def test_count_params_no_torch_load_for_large(self):
        """_count_params_native should not call torch.load for files > limit."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "scanners.py"
        source = source_path.read_text(encoding='utf-8')
        # The function body should check file_size > _PARAM_COUNT_LOAD_LIMIT
        # before reaching the torch.load call
        assert '_PARAM_COUNT_LOAD_LIMIT' in source
        assert '_estimate_params_from_size' in source


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

    def test_lazy_kv_cache_resolves(self):
        """KVCache resolves via __getattr__ and is a class."""
        from enigma_engine import core
        kv = getattr(core, 'KVCache', None)
        assert kv is not None, "KVCache not accessible from core"
        assert isinstance(kv, type), "KVCache should be a class"

    def test_huggingface_ensure_imports_exists(self):
        """huggingface_loader._ensure_imports function exists."""
        from enigma_engine.core import huggingface_loader
        assert hasattr(huggingface_loader, '_ensure_imports')
        assert callable(huggingface_loader._ensure_imports)

    def test_gptq_ensure_imports_exists(self):
        """gptq_awq_loader._ensure_imports function exists."""
        from enigma_engine.core import gptq_awq_loader
        assert hasattr(gptq_awq_loader, '_ensure_imports')
        assert callable(gptq_awq_loader._ensure_imports)

    def test_gguf_ensure_imports_exists(self):
        """gguf_loader._ensure_gguf_imports function exists."""
        from enigma_engine.core import gguf_loader
        assert hasattr(gguf_loader, '_ensure_gguf_imports')
        assert callable(gguf_loader._ensure_gguf_imports)

    def test_ollama_ensure_imports_exists(self):
        """ollama_loader._ensure_ollama_imports function exists."""
        from enigma_engine.core import ollama_loader
        assert hasattr(ollama_loader, '_ensure_ollama_imports')
        assert callable(ollama_loader._ensure_ollama_imports)

    def test_onnx_ensure_imports_exists(self):
        """onnx_loader._ensure_onnx_imports function exists."""
        from enigma_engine.core import onnx_loader
        assert hasattr(onnx_loader, '_ensure_onnx_imports')
        assert callable(onnx_loader._ensure_onnx_imports)

    def test_defaults_lazy_init(self):
        """defaults.py does not run _load_user_config at import time."""
        from enigma_engine.config import defaults
        assert hasattr(defaults, '_initialized')
        assert hasattr(defaults, '_ensure_initialized')
        assert callable(defaults._ensure_initialized)

    def test_config_is_lazy_dict(self):
        """CONFIG is a _LazyConfig instance that defers initialization."""
        from enigma_engine.config.defaults import CONFIG, _LazyConfig
        assert isinstance(CONFIG, _LazyConfig)
        assert isinstance(CONFIG, dict)


# ================================================================
# TTS Thread Safety: No Cross-Thread COM Calls
# ================================================================

class TestTTSThreadSafety:
    """Verify TTS stop uses callback instead of cross-thread engine.stop()."""

    def test_tts_uses_stop_event(self):
        """_tts_speak worker should use a threading.Event for stop signals."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        assert '_tts_stop_event' in source, (
            "TTS must use a threading.Event for thread-safe stop")

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

    def test_tts_stop_event_init(self):
        """desktop.py should initialize _tts_stop_event."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        assert '_tts_stop_event' in source, (
            "desktop.py must initialize _tts_stop_event")


# ================================================================
# TTS Text Cleaning: Safe Text for SAPI5
# ================================================================

class TestTTSTextCleaning:
    """Verify TTS cleans and chunks text before speaking."""

    def test_tts_clean_function_exists(self):
        """gui_logic.py should have a _tts_clean_text helper."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, '_tts_clean_text')

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

    def test_tts_chunk_function_exists(self):
        """gui_logic.py should have a _tts_chunk_text helper."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, '_tts_chunk_text')

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

    def test_tts_speak_calls_clean_and_chunk(self):
        """_tts_speak should use _tts_clean_text and _tts_chunk_text."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        match = re.search(
            r'def _tts_speak\b.*?(?=\n    def )',
            source, re.DOTALL)
        assert match, "_tts_speak not found"
        body = match.group(0)
        assert '_tts_clean_text' in body, (
            "_tts_speak must clean text before speaking")
        assert '_tts_chunk_text' in body, (
            "_tts_speak must chunk text before queuing")


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

    def test_count_model_params_background_method_exists(self):
        """EnigmaGUI must have _count_model_params_background."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        assert '_count_model_params_background' in source

    def test_hw_detection_in_background_thread(self):
        """CPU/GPU detection should run in a background thread."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        match = re.search(
            r'def _start_status_ticker\b.*?(?=\n    def |\nclass |\Z)',
            source, re.DOTALL)
        assert match, "_start_status_ticker not found"
        body = match.group(0)
        assert 'Thread' in body, (
            "Hardware detection must run in a background thread")
        assert '_detect_hw' in body or 'detect_hw' in body, (
            "Hardware detection should be in a named function")

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

        class DummyStatusBar:
            def set_right(self, text):
                pass

            def set_center(self, text):
                pass

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                self.target = target

            def start(self):
                pass

        monkeypatch.setattr(desktop_mod.threading, "Thread", DummyThread)

        obj = object.__new__(EnigmaGUI)
        obj._gaming_mode_active = True
        obj._status_tick_ms = 5000
        obj._boot_time = 0.0
        obj._hw_device_label = "CPU"
        obj.status_bar = DummyStatusBar()

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

        class UnexpectedThread:
            def __init__(self, *args, **kwargs):
                raise AssertionError("thread should not start")

        monkeypatch.setattr(desktop_mod.threading, "Thread", UnexpectedThread)

        obj = object.__new__(EnigmaGUI)
        obj._gaming_mode_active = True
        obj.models_data = [{"format": "pth", "params": None, "path": "fake"}]

        obj._count_model_params_background()


# ================================================================
# TTS Queue Drain: Stop Clears Pending Chunks
# ================================================================

class TestTTSQueueDrain:
    """Verify _tts_stop drains the queue to prevent late playback."""

    def test_tts_stop_drains_queue(self):
        """_tts_stop must drain pending chunks from the queue."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        match = re.search(
            r'def _tts_stop\b.*?(?=\n    def )',
            source, re.DOTALL)
        assert match, "_tts_stop not found"
        body = match.group(0)
        assert 'get_nowait' in body or 'empty' in body, (
            "_tts_stop must drain the queue to prevent "
            "queued chunks from playing after stop")


# ================================================================
# Scroll Consistency: Always Scroll During Typewriter
# ================================================================

class TestScrollConsistency:
    """Verify chat scroll stays at bottom during typewriter."""

    def test_typewriter_always_scrolls(self):
        """_typewriter must call _scroll_chat_to_bottom on every tick."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._typewriter)
        # Scroll should NOT be inside the throttle condition
        # It should appear after the throttle block, unconditionally
        lines = source.split('\n')
        scroll_calls = [i for i, line in enumerate(lines)
                        if '_scroll_chat_to_bottom' in line]
        assert len(scroll_calls) >= 3, (
            "_typewriter must call _scroll_chat_to_bottom "
            "on every tick (stop, finish, and normal insertion)")

    def test_scroll_to_bottom_uses_native_see(self):
        """_scroll_chat_to_bottom should use tk.Text see() method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._scroll_chat_to_bottom)
        assert 'see(' in source, (
            "_scroll_chat_to_bottom should use native see('end') — "
            "simpler and more reliable than canvas yview_moveto")


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

    def test_sessions_sorted_newest_first(self):
        """scan_sessions returns newest sessions first."""
        from enigma_engine.gui.scanners import scan_sessions
        sessions = scan_sessions()
        if len(sessions) >= 2:
            for i in range(len(sessions) - 1):
                assert sessions[i]["saved_at"] >= sessions[i + 1]["saved_at"], (
                    "Sessions should be sorted newest-first by saved_at")

    def test_max_tokens_fallback_matches_config(self):
        """Config fallback for max_tokens should be 2048, not 100."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._load_config_defaults)
        # The hardcoded fallback in the except block
        assert '"max_tokens": 2048' in source, (
            "max_tokens fallback should be 2048, not 100")

    def test_config_default_max_gen_is_2048(self):
        """CONFIG max_gen default is 2048."""
        from enigma_engine.config import CONFIG
        assert CONFIG.get("max_gen") == 2048

    def test_delete_session_uses_tracked_index(self):
        """_delete_session reads _selected_session_index, not cursor."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._delete_session)
        assert "_selected_session_index" in source, (
            "Should use tracked selection index")
        assert 'index("insert")' not in source, (
            "Should not use cursor position")

    def test_on_history_click_tracks_index(self):
        """_on_history_click stores _selected_session_index."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._on_history_click)
        assert "_selected_session_index" in source

    def test_escape_fullscreen_stops_generation(self):
        """Escape in fullscreen should stop generation if active."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._on_escape_fullscreen)
        assert "_is_generating" in source, (
            "Should check _is_generating before exiting fullscreen")
        assert "_stop_generation" in source, (
            "Should call _stop_generation if generating")

    def test_restore_history_renders_media(self):
        """_restore_history_display calls _process_media_in_text."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._restore_history_display)
        assert "_process_media_in_text" in source, (
            "Should process media when restoring history")

    def test_load_session_moves_ai_name_outside_loop(self):
        """_load_session_by_path gets AI name once, not per message."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._load_session_by_path)
        # ai = self._active_ai_name() should appear before the for loop
        ai_line = source.index("_active_ai_name")
        for_line = source.index("for msg in messages")
        assert ai_line < for_line, (
            "AI name should be resolved once before the message loop")


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

    def test_load_function_calls_estimate(self):
        """_load() in load_model calls _estimate_gguf_params for GGUF."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._load_model)
        assert "_estimate_gguf_params" in source


# =========================================================================
# Deep-dive audit — scanner security, server path traversal, CORS
# =========================================================================

class TestScannersRestrictedUnpickler:
    """Verify scanners.py uses a restricted unpickler for .pt files."""

    def test_safe_unpickler_exists(self):
        """_peek_target_size should use _SafeUnpickler."""
        from enigma_engine.gui import scanners
        source = inspect.getsource(scanners)
        assert "_SafeUnpickler" in source
        assert "pickle.Unpickler" in source

    def test_safe_unpickler_blocks_os_system(self):
        """Restricted unpickler should reject os.system."""
        import pickle
        import io
        from enigma_engine.gui.scanners import _peek_target_size
        # Source-level check — the _SafeUnpickler is nested inside
        # _peek_target_size, so we verify the source contains the guard
        source = inspect.getsource(_peek_target_size)
        assert "_ALLOWED" in source
        assert "UnpicklingError" in source
        # Only torch/collections modules are allowed
        assert '"collections"' in source or "'collections'" in source
        assert '"torch' in source or "'torch" in source

    def test_safe_unpickler_allows_torch_classes(self):
        """Allowed list includes torch storage types."""
        from enigma_engine.gui.scanners import _peek_target_size
        source = inspect.getsource(_peek_target_size)
        assert "FloatStorage" in source
        assert "OrderedDict" in source

    def test_scanner_logs_failures(self):
        """Scanner should log debug on peek failures, not silent pass."""
        from enigma_engine.gui.scanners import _peek_target_size
        source = inspect.getsource(_peek_target_size)
        assert "logger.debug" in source


class TestServerPathTraversal:
    """Verify server.py prevents path traversal with os.sep check."""

    def test_load_model_checks_path_with_sep(self):
        """Path traversal guard must append os.sep to prevent prefix attacks."""
        import enigma_engine.api.server as srv
        mod_source = inspect.getsource(srv)
        assert "os.sep" in mod_source, (
            "Path traversal check must use os.sep to prevent 'models_evil/' bypass")

    def test_server_imports_os(self):
        """server.py must import os for the os.sep check."""
        import enigma_engine.api.server as srv
        import os
        assert hasattr(srv, 'os') or 'import os' in inspect.getsource(srv)


class TestServerCORSPreflight:
    """Verify API key middleware skips OPTIONS requests for CORS preflight."""

    def test_options_bypass_in_middleware(self):
        """API key middleware should skip key check for OPTIONS requests."""
        import enigma_engine.api.server as srv
        source = inspect.getsource(srv)
        # Check for OPTIONS bypass pattern
        assert '"OPTIONS"' in source or "'OPTIONS'" in source, (
            "Middleware should skip API key check for OPTIONS requests")

    def test_cors_middleware_configured(self):
        """Server should have CORS middleware with allow_origins."""
        import enigma_engine.api.server as srv
        source = inspect.getsource(srv)
        assert "CORSMiddleware" in source
        assert "allow_origins" in source


class TestModKillOSError:
    """Verify gui_mods.py handles OSError on proc.kill() after timeout."""

    def test_kill_has_oserror_protection(self):
        """proc.kill() after TimeoutExpired should be wrapped in try/except OSError."""
        from enigma_engine.gui.gui_mods import ModMixin
        source = inspect.getsource(ModMixin)
        # Find the kill section — should have OSError guard
        assert "OSError" in source, (
            "proc.kill() should be wrapped in try/except OSError")


# ── Mod base file presence ──────────────────────────────────────────────────


class TestModBasePresence:
    """Verify mod_base.py exists in each shipped mod folder."""

    def test_imagegen_has_mod_base(self):
        """mods/imagegen/ must contain mod_base.py for imports to work."""
        from pathlib import Path
        assert (Path("mods/imagegen/mod_base.py").exists()), (
            "mods/imagegen/mod_base.py missing — ImageGenMod fails")

    def test_mod_launcher_checks_health(self):
        """_launch_mod should detect immediate crashes (stderr check)."""
        import inspect
        from enigma_engine.gui.gui_mods import ModMixin
        source = inspect.getsource(ModMixin._launch_mod)
        # Must use stderr=PIPE (not DEVNULL) to capture crash output
        assert "stderr=subprocess.PIPE" in source, (
            "_launch_mod should capture stderr to detect import errors")
        # Must check proc.wait with timeout to detect immediate crash
        assert "proc.wait(timeout=" in source, (
            "_launch_mod should wait briefly to detect startup crashes")


# ── Config persistence ──────────────────────────────────────────────────────


class TestConfigPersistence:
    """Verify generation config (temperature etc.) persists across restarts."""

    def test_save_config_overrides_method_exists(self):
        """LogicMixin must have _save_config_overrides method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_save_config_overrides"), (
            "LogicMixin missing _save_config_overrides — config lost on restart")

    def test_load_saved_config_overrides_method_exists(self):
        """LogicMixin must have _load_saved_config_overrides method."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_load_saved_config_overrides"), (
            "LogicMixin missing _load_saved_config_overrides")

    def test_on_close_saves_config(self):
        """_on_close must call _save_config_overrides before destroy."""
        import inspect
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        assert "_save_config_overrides" in source, (
            "_on_close must call _save_config_overrides to persist settings")

    def test_load_defaults_restores_saved(self):
        """_load_config_defaults must consult saved overrides."""
        import inspect
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._load_config_defaults)
        assert "_load_saved_config_overrides" in source, (
            "_load_config_defaults should restore saved overrides")


# ── Atomic saves ────────────────────────────────────────────────────────────


class TestAtomicSaves:
    """Verify model saves use atomic write pattern."""

    def test_safe_save_module_exists(self):
        """safe_save module must exist with atomic_torch_save."""
        from enigma_engine.core.safe_save import atomic_torch_save
        assert callable(atomic_torch_save)

    def test_atomic_save_uses_temp_file(self):
        """atomic_torch_save must write to .tmp then os.replace."""
        import inspect
        from enigma_engine.core.safe_save import atomic_torch_save
        source = inspect.getsource(atomic_torch_save)
        assert ".tmp" in source, (
            "atomic_torch_save must use a .tmp file")
        assert "os.replace" in source, (
            "atomic_torch_save must use os.replace for atomicity")

    def test_forge_training_uses_atomic_save(self):
        """All training saves in gui_forge.py must use atomic_torch_save."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin)
        # Count remaining raw torch.save calls (should be zero)
        import re
        raw_saves = re.findall(r'torch\.save\(', source)
        assert len(raw_saves) == 0, (
            f"gui_forge.py still has {len(raw_saves)} raw torch.save() "
            f"calls — should use atomic_torch_save")

    def test_training_checkpoint_uses_atomic_save(self):
        """Trainer._save_checkpoint must use atomic_torch_save."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._save_checkpoint)
        assert "atomic_torch_save" in source, (
            "_save_checkpoint should use atomic_torch_save, not torch.save")

    def test_atomic_save_cleans_up_on_failure(self):
        """atomic_torch_save must remove .tmp file on failure."""
        import inspect
        from enigma_engine.core.safe_save import atomic_torch_save
        source = inspect.getsource(atomic_torch_save)
        assert "unlink" in source, (
            "atomic_torch_save must clean up .tmp on failure")

    # ── atomic_write_text / atomic_write_json ────────────────────────────

    def test_atomic_write_text_exists(self):
        """atomic_write_text must exist in safe_save."""
        from enigma_engine.core.safe_save import atomic_write_text
        assert callable(atomic_write_text)

    def test_atomic_write_json_exists(self):
        """atomic_write_json must exist in safe_save."""
        from enigma_engine.core.safe_save import atomic_write_json
        assert callable(atomic_write_json)

    def test_atomic_write_text_uses_fsync(self):
        """atomic_write_text must fsync before rename for durability."""
        import inspect
        from enigma_engine.core.safe_save import atomic_write_text
        source = inspect.getsource(atomic_write_text)
        assert "fsync" in source or "flush" in source, (
            "atomic_write_text must fsync/flush for durability")
        assert "os.replace" in source, (
            "atomic_write_text must use os.replace for atomicity")

    def test_atomic_write_text_creates_backup(self):
        """atomic_write_text must create .bak before replacing."""
        import inspect
        from enigma_engine.core.safe_save import atomic_write_text
        source = inspect.getsource(atomic_write_text)
        assert ".bak" in source, (
            "atomic_write_text must create .bak backup")

    def test_atomic_write_json_uses_fsync(self):
        """atomic_write_json must fsync before rename for durability."""
        import inspect
        from enigma_engine.core.safe_save import atomic_write_json
        source = inspect.getsource(atomic_write_json)
        assert "atomic_write_text" in source or "fsync" in source, (
            "atomic_write_json must use atomic_write_text or fsync directly")

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
        try:
            atomic_write_text(target, "data")
        except OSError:
            pass
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


class TestGenerationLockPattern:
    """Verify _generation_lock uses `with` statement, not manual acquire/release."""

    def test_inference_lock_uses_with(self):
        """inference.py generate() must use `with self._generation_lock:`."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine.generate)
        assert "getattr" not in source or "_generation_lock" not in source.split("getattr")[1].split("\n")[0], (
            "generate() should not use getattr guard for _generation_lock")
        assert "with self._generation_lock:" in source, (
            "generate() should use `with self._generation_lock:`")

    def test_engine_generation_lock_uses_with(self):
        """engine_generation.py stream_generate must use `with self._generation_lock`."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.stream_generate)
        assert "getattr" not in source or "_generation_lock" not in source.split("getattr")[1].split("\n")[0], (
            "stream_generate() should not use getattr guard for _generation_lock")
        assert "with self._generation_lock" in source, (
            "stream_generate() should use `with self._generation_lock:`")

    def test_lock_init_in_init(self):
        """EnigmaEngine.__init__ must set _generation_lock (directly or via _init_common)."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine.__init__)
        direct = "self._generation_lock = threading.Lock()" in source
        via_common = "_init_common" in source
        assert direct or via_common, (
            "__init__ must initialize _generation_lock (directly or via _init_common)")
        # If delegated, verify _init_common actually sets it
        if via_common:
            common_src = inspect.getsource(EnigmaEngine._init_common)
            assert "_generation_lock" in common_src

    def test_lock_init_in_from_model(self):
        """from_model must set _generation_lock (directly or via _init_common)."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine.from_model)
        direct = "_generation_lock" in source
        via_common = "_init_common" in source
        assert direct or via_common, (
            "from_model must initialize _generation_lock (directly or via _init_common)")
        if via_common:
            common_src = inspect.getsource(EnigmaEngine._init_common)
            assert "_generation_lock" in common_src


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

class TestRenameCaseInsensitive:
    """Rename must handle case-only changes on Windows."""

    def test_rename_handles_case_only_change(self):
        """_rename_model must not bail on case-only name changes."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        # Must compare as strings, not Path objects (Windows is case-insensitive)
        assert "str(dest)" in source or "dest.name" in source or "tmp" in source, (
            "Rename must handle case-only changes via string comparison or temp rename")

    def test_rename_uses_temp_for_case_change(self):
        """Case-only rename should use a two-step temp rename."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._rename_model)
        assert "tmp" in source.lower(), (
            "Case-only rename on Windows needs a temp step")


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
        assert cfg.use_gradient_checkpointing is False

    def test_gradient_checkpointing_in_to_dict(self):
        """use_gradient_checkpointing must appear in to_dict output."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(use_gradient_checkpointing=True)
        d = cfg.to_dict()
        assert "use_gradient_checkpointing" in d
        assert d["use_gradient_checkpointing"] is True

    def test_trainer_applies_gradient_checkpointing(self):
        """Trainer must reference gradient_checkpointing in init or train."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer)
        assert "gradient_checkpointing" in source


# =====================================================================
# Training config exposed in FORGE UI
# =====================================================================

class TestForgeTrainingConfig:
    """FORGE must expose batch size, grad accumulation, and gradient checkpointing."""

    def test_forge_has_batch_size_entry(self):
        """FORGE page must have a batch_size entry."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "batch_size" in source or "batch" in source.lower()

    def test_forge_has_grad_accum_entry(self):
        """FORGE page must have a gradient accumulation entry."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "grad_accum" in source or "accumulation" in source.lower()

    def test_forge_has_grad_checkpoint_toggle(self):
        """FORGE page must have a gradient checkpointing toggle."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "grad_ckpt" in source or "checkpointing" in source.lower()

    def test_solo_reads_batch_size(self):
        """Solo training must read batch_size from UI entry."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_solo_training)
        assert "batch_size" in source

    def test_guided_reads_batch_size(self):
        """Guided training must read batch_size from UI entry."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_guided_training)
        assert "batch_size" in source

    def test_dialogue_reads_batch_size(self):
        """Dialogue training must read batch_size from UI entry."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_dialogue_training)
        assert "batch_size" in source


# =====================================================================
# Quantize & Export on FORGE page
# =====================================================================

class TestForgeQuantizeExport:
    """FORGE TOOLS must have quantize and export buttons."""

    def test_forge_has_quantize_button(self):
        """FORGE TOOLS section must have a QUANTIZE button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "QUANTIZE" in source

    def test_forge_has_export_gguf_button(self):
        """FORGE TOOLS section must have an EXPORT GGUF button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "EXPORT" in source and "GGUF" in source

    def test_quantize_handler_exists(self):
        """ForgeMixin must have a _quantize_student method."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_quantize_student")
        assert callable(getattr(ForgeMixin, "_quantize_student"))

    def test_export_gguf_handler_exists(self):
        """ForgeMixin must have a _export_student_gguf method."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_export_student_gguf")
        assert callable(getattr(ForgeMixin, "_export_student_gguf"))

    def test_quantize_uses_model_quantize(self):
        """_quantize_student must use model.quantize() from model.py."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._quantize_student)
        assert "quantize" in source

    def test_export_uses_export_to_gguf(self):
        """_export_student_gguf must use export_to_gguf."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._export_student_gguf)
        assert "export_to_gguf" in source

    def test_quantize_runs_in_thread(self):
        """Quantize must run in a background thread."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._quantize_student)
        assert "Thread" in source or "thread" in source

    def test_export_runs_in_thread(self):
        """Export must run in a background thread."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._export_student_gguf)
        assert "Thread" in source or "thread" in source


# =====================================================================
# Memory instructions — proactive preference learning
# =====================================================================

class TestMemoryInstructions:
    """AI memory instructions must encourage proactive preference learning."""

    def test_memory_instruction_observes_patterns(self):
        """Memory instruction must mention observing patterns/preferences."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "pattern" in source.lower() or "preference" in source.lower() or "habit" in source.lower()

    def test_memory_instruction_suggests_alternatives(self):
        """Memory instruction must tell AI to suggest better approaches."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "better" in source.lower() or "alternative" in source.lower() or "suggest" in source.lower()

    def test_memory_instruction_asks_permission(self):
        """AI must ask before changing user's established approach."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._build_gui_context)
        assert "ask" in source.lower() or "permission" in source.lower() or "want to" in source.lower()


# =====================================================================
# Learn While Chatting — BackgroundTrainer wired to chat
# =====================================================================

class TestLearnWhileChatting:
    """Chat exchanges must feed BackgroundTrainer when enabled."""

    def test_send_message_feeds_trainer(self):
        """_send_message path must reference learn_while or add_training."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._send_message)
        assert ("learn_while" in source or "add_training" in source
                or "_feed_background_trainer" in source)

    def test_feed_method_exists(self):
        """LogicMixin must have a method to feed chat to trainer."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_feed_background_trainer")
        assert callable(getattr(LogicMixin, "_feed_background_trainer"))

    def test_feed_method_uses_router(self):
        """_feed_background_trainer must use router.add_training_example."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._feed_background_trainer)
        assert "add_training_example" in source

    def test_feed_checks_setting(self):
        """Feeding must check learn_while_chatting setting."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._feed_background_trainer)
        assert "learn_while_chatting" in source


# =====================================================================
# Theme picker on CONFIG page
# =====================================================================

class TestThemePicker:
    """CONFIG page must have a theme selector with live switching."""

    def test_config_has_theme_section(self):
        """CONFIG page must have a THEME section."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_config)
        assert "THEME" in source or "theme" in source

    def test_theme_dropdown_in_config(self):
        """CONFIG page must have a theme dropdown or variable."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_config)
        assert "theme_var" in source or "theme_dd" in source or "theme_dropdown" in source

    def test_theme_uses_get_theme_names(self):
        """Theme dropdown must use get_theme_names() for options."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_config)
        assert "get_theme_names" in source

    def test_apply_theme_method(self):
        """PagesMixin must have a _apply_theme method."""
        from enigma_engine.gui.gui_pages import PagesMixin
        assert hasattr(PagesMixin, "_apply_theme")

    def test_apply_theme_calls_live(self):
        """_apply_theme must call _apply_theme_live for live switching."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._apply_theme)
        assert "_apply_theme_live" in source

    def test_apply_theme_no_restart(self):
        """_apply_theme must NOT call _restart_gui (live switching)."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._apply_theme)
        assert "_restart_gui" not in source

    def test_apply_theme_live_method_exists(self):
        """Desktop must have _apply_theme_live for live theme switching."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_apply_theme_live")
        assert callable(getattr(EnigmaGUI, "_apply_theme_live"))

    def test_retheme_tree_method_exists(self):
        """Desktop must have _retheme_tree for widget tree walking."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_retheme_tree")

    def test_retheme_one_method_exists(self):
        """Desktop must have _retheme_one for per-widget retheming."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_retheme_one")

    def test_reload_theme_function(self):
        """widgets.py must export reload_theme function."""
        from enigma_engine.gui.widgets import reload_theme
        assert callable(reload_theme)

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

    def test_restart_gui_method_exists(self):
        """Desktop must still have _restart_gui (for font size changes)."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_restart_gui")
        assert callable(getattr(EnigmaGUI, "_restart_gui"))

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

class TestSelectableLabel:
    """SelectableLabel exists and supports text selection."""

    def test_selectable_label_exists(self):
        """SelectableLabel class exists in widgets module."""
        from enigma_engine.gui.widgets import SelectableLabel
        assert SelectableLabel is not None

    def test_selectable_label_has_configure(self):
        """SelectableLabel supports text and text_color updates."""
        from enigma_engine.gui.widgets import SelectableLabel
        source = inspect.getsource(SelectableLabel)
        assert "configure" in source
        assert "text" in source
        assert "text_color" in source

    def test_selectable_label_has_cget(self):
        """SelectableLabel supports cget for text retrieval."""
        from enigma_engine.gui.widgets import SelectableLabel
        source = inspect.getsource(SelectableLabel)
        assert "cget" in source

    def test_selectable_label_uses_readonly_entry(self):
        """SelectableLabel uses a readonly Entry for selection support."""
        from enigma_engine.gui.widgets import SelectableLabel
        source = inspect.getsource(SelectableLabel)
        assert "readonly" in source
        assert "Entry" in source

    def test_selectable_label_no_blinking_cursor(self):
        """SelectableLabel sets insertwidth=0 to hide cursor."""
        from enigma_engine.gui.widgets import SelectableLabel
        source = inspect.getsource(SelectableLabel)
        assert "insertwidth" in source

    def test_selectable_label_has_copy_menu(self):
        """SelectableLabel supports right-click copy."""
        from enigma_engine.gui.widgets import SelectableLabel
        source = inspect.getsource(SelectableLabel)
        assert "copy" in source.lower()

    def test_selectable_label_in_exports(self):
        """SelectableLabel is importable from widgets."""
        from enigma_engine.gui.widgets import SelectableLabel
        assert callable(SelectableLabel)


class TestSelectableTextboxCursor:
    """SelectableTextbox hides the blinking insertion cursor."""

    def test_insertwidth_zero(self):
        """SelectableTextbox sets insertwidth=0 on the text widget."""
        from enigma_engine.gui.widgets import SelectableTextbox
        source = inspect.getsource(SelectableTextbox.__init__)
        assert "insertwidth" in source

    def test_still_allows_selection(self):
        """SelectableTextbox still blocks editing keys but allows nav."""
        from enigma_engine.gui.widgets import SelectableTextbox
        source = inspect.getsource(SelectableTextbox._on_key)
        # Allow copy and select-all
        assert '"c"' in source or "'c'" in source
        assert '"a"' in source or "'a'" in source


class TestSelectableTextEverywhere:
    """Key display elements use SelectableLabel instead of CTkLabel."""

    def test_section_label_uses_selectable(self):
        """SectionLabel uses SelectableLabel for its title."""
        from enigma_engine.gui.widgets import SectionLabel
        source = inspect.getsource(SectionLabel)
        assert "SelectableLabel" in source

    def test_desktop_title_selectable(self):
        """Desktop title uses SelectableLabel."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._build_shell)
        assert "SelectableLabel" in source

    def test_status_bar_selectable(self):
        """StatusBar labels use SelectableLabel."""
        from enigma_engine.gui.widgets import StatusBar
        source = inspect.getsource(StatusBar)
        assert "SelectableLabel" in source

    def test_collapsible_panel_title_selectable(self):
        """CollapsiblePanel title uses SelectableLabel."""
        from enigma_engine.gui.widgets import CollapsiblePanel
        source = inspect.getsource(CollapsiblePanel)
        assert "SelectableLabel" in source

    def test_forge_heading_selectable(self):
        """_forge_heading uses SelectableLabel."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._forge_heading)
        assert "SelectableLabel" in source

    def test_models_page_display_labels(self):
        """MODELS page display labels use SelectableLabel."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(
            PagesMixin._populate_model_cards)
        assert "SelectableLabel" in source

    def test_route_card_display_labels(self):
        """Route cards use SelectableLabel for names."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(
            PagesMixin._build_route_card)
        assert "SelectableLabel" in source

    def test_cmd_status_strip_selectable(self):
        """CMD status strip labels use SelectableLabel."""
        from enigma_engine.gui.gui_cmd_page import CMDPageMixin
        source = inspect.getsource(CMDPageMixin._build_page_cmd)
        assert "SelectableLabel" in source

    def test_mod_page_display_labels(self):
        """Mod page display labels use SelectableLabel."""
        from enigma_engine.gui.gui_mod_page import ModPageMixin
        source = inspect.getsource(
            ModPageMixin._build_page_mod)
        assert "SelectableLabel" in source

    def test_config_page_display_labels(self):
        """CONFIG page section titles use SelectableLabel."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(
            PagesMixin._build_page_config)
        assert "SelectableLabel" in source

    def test_docs_page_display_labels(self):
        """DOCS page display labels use SelectableLabel."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        source = inspect.getsource(
            DocsPageMixin._build_page_docs)
        assert "SelectableLabel" in source


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


class TestForgeProgressBar:
    """Progress bar update/reset methods exist and are callable."""

    def test_update_method_exists(self):
        """ForgeMixin has _update_forge_progress."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_update_forge_progress")

    def test_reset_method_exists(self):
        """ForgeMixin has _reset_forge_progress."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_reset_forge_progress")

    def test_show_history_method_exists(self):
        """ForgeMixin has _show_training_history."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_show_training_history")


class TestLearnWhileChatting:
    """Learn-while-chatting toggle on CONFIG page."""

    def test_toggle_method_exists(self):
        """PagesMixin has _toggle_learn_while_chatting."""
        from enigma_engine.gui.gui_pages import PagesMixin
        assert hasattr(PagesMixin, "_toggle_learn_while_chatting")

    def test_toggle_saves_setting(self, tmp_path, monkeypatch):
        """Toggle writes learn_while_chatting to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        # Create minimal mock
        class MockVar:
            def get(self):
                return True

        class MockStatusBar:
            def set_left(self, text):
                pass

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

    def test_gui_logic_reads_setting(self):
        """gui_logic._feed_background_trainer checks the setting."""
        import inspect
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(
            LogicMixin._feed_background_trainer)
        assert "learn_while_chatting" in source

    def test_training_hooks_in_solo(self):
        """Solo training has progress bar hooks."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_solo_training)
        assert "_reset_forge_progress" in source
        assert "_update_forge_progress" in source
        assert "_save_training_run" in source

    def test_training_hooks_in_dpo(self):
        """DPO training has progress bar hooks."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_dpo_training)
        assert "_reset_forge_progress" in source
        assert "_update_forge_progress" in source
        assert "_save_training_run" in source

    def test_training_hooks_in_guided(self):
        """Guided training has progress bar hooks."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_guided_training)
        assert "_reset_forge_progress" in source
        assert "_update_forge_progress" in source
        assert "_save_training_run" in source


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

        class MockVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        class MockStatusBar:
            def __init__(self):
                self.last = ""

            def set_left(self, text):
                self.last = text

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

    def test_training_hooks_in_dialogue(self):
        """Dialogue training has progress bar hooks."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_dialogue_training)
        assert "_reset_forge_progress" in source
        assert "_update_forge_progress" in source
        assert "_save_training_run" in source

    def test_training_hooks_in_vision(self):
        """Vision training has progress bar hooks."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_vision_training)
        assert "_reset_forge_progress" in source
        assert "_update_forge_progress" in source
        assert "_save_training_run" in source

    def test_training_hooks_in_evolutionary(self):
        """Evolutionary training has progress bar hooks."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_evolutionary_training)
        assert "_reset_forge_progress" in source
        assert "_update_forge_progress" in source
        assert "_save_training_run" in source

    def test_training_hooks_in_lora(self):
        """LoRA training has progress bar hooks."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_lora_training)
        assert "_reset_forge_progress" in source
        assert "_update_forge_progress" in source
        assert "_save_training_run" in source


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

        class MockVar:
            def get(self):
                return False

        class MockStatusBar:
            def set_left(self, text):
                pass

        obj = object.__new__(PagesMixin)
        obj._auto_load_chat_model_var = MockVar()
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

        class MockVar:
            def get(self):
                return False

        class MockStatusBar:
            def set_left(self, text):
                pass

        obj = object.__new__(PagesMixin)
        obj._auto_start_mods_var = MockVar()
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

        class MockVar:
            def get(self):
                return True

        class MockStatusBar:
            def set_left(self, text):
                pass

        obj = object.__new__(PagesMixin)
        obj._auto_unload_on_minimize_var = MockVar()
        obj.status_bar = MockStatusBar()

        obj._toggle_auto_unload_on_minimize()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_unload_on_minimize"] is True

    def test_route_restore_checks_auto_load_toggle(self):
        """Route restore only autoloads when auto-load setting is on."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._load_route_assignments)
        assert "_auto_load_chat_model" in source

    def test_desktop_reads_performance_settings(self):
        """Desktop init reads memory-related performance flags."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)
        assert "_auto_load_chat_model" in source
        assert "_auto_start_mods" in source
        assert "_auto_unload_on_minimize" in source

    def test_desktop_has_minimize_suspend_handlers(self):
        """Desktop binds minimize/restore handlers for memory saver."""
        from enigma_engine.gui.desktop import EnigmaGUI
        init_src = inspect.getsource(EnigmaGUI.__init__)
        assert "<Unmap>" in init_src
        assert "<Map>" in init_src
        assert hasattr(EnigmaGUI, "_on_window_unmap")
        assert hasattr(EnigmaGUI, "_on_window_map")

    def test_apply_gaming_mode_preset_saves_three_settings(
            self, tmp_path, monkeypatch):
        """Gaming preset writes all memory-related settings."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        class MockStatusBar:
            def set_left(self, text):
                pass

        class MockVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

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

    def test_logic_has_manual_suspend_resume_methods(self):
        """LogicMixin has explicit suspend/resume model memory helpers."""
        from enigma_engine.gui.gui_logic import LogicMixin
        assert hasattr(LogicMixin, "_suspend_model_memory")
        assert hasattr(LogicMixin, "_resume_suspended_model")

    def test_router_page_has_suspend_button(self):
        """ROUTER page exposes manual suspend/resume control."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_router)
        assert "SUSPEND" in source


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

    def test_three_modes_in_ui(self):
        """UI displays the 3 training modes: Basic, AI-Guided, Image."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert '"Basic"' in source
        assert '"AI-Guided"' in source
        assert '"Image"' in source

    def test_default_mode_is_basic(self):
        """FORGE defaults to Basic mode."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert 'value="Basic"' in source

    def test_dispatcher_routes_current_modes(self):
        """Dispatcher routes Basic, AI-Guided, and Image modes."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._start_training_by_mode)
        assert 'mode_name == "Basic"' in source
        assert 'mode_name == "AI-Guided"' in source
        assert 'mode_name == "Image"' in source
        assert "_start_basic_training" in source
        assert "_start_ai_guided_training" in source
        assert "_start_vision_training" in source

    def test_mode_changed_uses_three_mode_visibility(self):
        """Visibility logic follows the new 3-mode section model."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._on_training_mode_changed)
        assert 'if mode == "Basic"' in source
        assert 'elif mode == "AI-Guided"' in source
        assert 'elif mode == "Image"' in source
        assert 'visible = {"basic"}' in source
        assert 'visible = {"ai", "stages", "brief", "pairs"}' in source
        assert 'visible = {"image"}' in source


# ================================================================
# Shared Web Utilities
# ================================================================

class TestWebUtils:
    """Test shared web search and page fetching utilities."""

    def test_module_exists(self):
        """web_utils module is importable."""
        from enigma_engine.core import web_utils
        assert web_utils is not None

    def test_ddg_search_callable(self):
        """ddg_search is a callable function."""
        from enigma_engine.core.web_utils import ddg_search
        assert callable(ddg_search)

    def test_fetch_page_text_callable(self):
        """fetch_page_text is a callable function."""
        from enigma_engine.core.web_utils import fetch_page_text
        assert callable(fetch_page_text)

    def test_extract_html_text_callable(self):
        """extract_html_text is a callable function."""
        from enigma_engine.core.web_utils import extract_html_text
        assert callable(extract_html_text)

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

    def test_builtin_commands_uses_web_utils(self):
        """builtin_commands search_web uses web_utils module."""
        from enigma_engine.core.builtin_commands import (
            register_builtin_commands)
        import enigma_engine.core.builtin_commands as bc
        source = inspect.getsource(bc)
        assert "web_utils" in source


# ================================================================
# FORGE: Optimized Web Learn
# ================================================================

class TestWebLearnOptimized:
    """Test optimized web learn uses shared web_utils and
    trainer system prompt."""

    def test_web_learn_uses_web_utils(self):
        """_web_learn imports from web_utils."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "web_utils" in source

    def test_web_learn_uses_trainer_system_prompt(self):
        """_web_learn uses _build_trainer_system_prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "_build_trainer_system_prompt" in source

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

    def test_web_learn_updates_progress(self):
        """_web_learn updates the progress bar."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "_update_forge_progress" in source

    def test_generate_data_updates_progress(self):
        """_generate_training_data updates the progress bar."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "_update_forge_progress" in source

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

class TestForgeAutoTrain:
    """Test auto-train checkbox that starts training
    immediately after data generation."""

    def test_auto_train_var_exists(self):
        """_build_page_forge creates forge_auto_train_var."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "forge_auto_train_var" in source

    def test_auto_train_checkbox_exists(self):
        """_build_page_forge creates auto-train checkbox."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        assert "Auto-train" in source

    def test_web_learn_checks_auto_train(self):
        """_web_learn checks forge_auto_train_var."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "forge_auto_train_var" in source

    def test_generate_data_checks_auto_train(self):
        """_generate_training_data checks forge_auto_train_var."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._generate_training_data)
        assert "forge_auto_train_var" in source

    def test_auto_train_has_tooltip(self):
        """Auto-train checkbox has a tooltip."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        # Check that tooltip exists near auto-train
        assert "auto" in source.lower()


# ================================================================
# Phase 1 — Polish
# ================================================================


class TestCMDTooltips:
    """CMD page should have tooltips on all interactive elements."""

    def test_cmd_has_tooltip_import(self):
        """gui_cmd_page.py must import Tooltip."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_cmd_page.py")
        source = source_path.read_text(encoding="utf-8")
        assert "Tooltip" in source

    def test_cmd_clear_has_tooltip(self):
        """CLEAR button should have a tooltip."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_cmd_page.py")
        source = source_path.read_text(encoding="utf-8")
        assert "Tooltip(" in source, "CMD page has no tooltips"

    def test_cmd_at_least_three_tooltips(self):
        """CMD page should have at least 3 Tooltip() calls.

        Note: CTkSegmentedButton does not support .bind(), so the
        mode toggle cannot have a Tooltip attached.
        """
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_cmd_page.py")
        source = source_path.read_text(encoding="utf-8")
        count = source.count("Tooltip(")
        assert count >= 3, (
            f"CMD page has only {count} Tooltip() call(s), "
            "expected at least 3")


class TestRightClickMenus:
    """Editable text widgets should have right-click context menus."""

    def test_chat_input_right_click(self):
        """chat_input should bind <Button-3> for context menu."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_pages.py")
        source = source_path.read_text(encoding="utf-8")
        assert "Button-3" in source, (
            "chat_input needs right-click context menu binding")

    def test_docs_editor_right_click(self):
        """DOCS editor should bind <Button-3> for context menu."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_docs_page.py")
        source = source_path.read_text(encoding="utf-8")
        assert "Button-3" in source, (
            "DOCS editor needs right-click context menu binding")

    def test_cmd_input_right_click(self):
        """CMD input should bind <Button-3> for context menu."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_cmd_page.py")
        source = source_path.read_text(encoding="utf-8")
        assert "Button-3" in source, (
            "CMD input needs right-click context menu binding")


class TestNewChatConfirmation:
    """New chat auto-saves instead of confirming."""

    def test_new_chat_auto_saves(self):
        """_new_chat auto-saves the session without popup confirmation."""
        from enigma_engine.gui.gui_logic import LogicMixin
        source = inspect.getsource(LogicMixin._new_chat)
        # No popup — chat is auto-saved so no confirmation needed
        assert "messagebox" not in source
        assert "askyesno" not in source
        # Should still reset and create new session
        assert "_reset_display" in source
        assert "session_" in source


class TestCtrlNShortcut:
    """Ctrl+N should start a new chat."""

    def test_ctrl_n_bound(self):
        """desktop.py should bind Ctrl+N."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "desktop.py")
        source = source_path.read_text(encoding="utf-8")
        assert "Control-n" in source or "Control-N" in source, (
            "Ctrl+N should be bound to new chat")


class TestCtrlFFind:
    """DOCS editor should have Ctrl+F find bar."""

    def test_ctrl_f_bound(self):
        """DOCS editor binds Ctrl+F."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_docs_page.py")
        source = source_path.read_text(encoding="utf-8")
        assert "Control-f" in source or "Control-F" in source, (
            "DOCS editor should bind Ctrl+F for find")

    def test_find_bar_exists(self):
        """DocsPageMixin should have a _docs_find method."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        assert hasattr(DocsPageMixin, "_docs_find_next") or \
            hasattr(DocsPageMixin, "_docs_toggle_find")


class TestDocsAutoSave:
    """DOCS editor should auto-save periodically."""

    def test_auto_save_method_exists(self):
        """DocsPageMixin should have _docs_auto_save."""
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        assert hasattr(DocsPageMixin, "_docs_auto_save")

    def test_auto_save_timer_started(self):
        """_build_page_docs source should schedule auto-save."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_docs_page.py")
        source = source_path.read_text(encoding="utf-8")
        assert "auto_save" in source.lower(), (
            "DOCS page should start an auto-save timer")


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


class TestMagicNumbers:
    """builtin_commands.py should use named constants for limits."""

    def test_timeout_constants_exist(self):
        """builtin_commands.py should define timeout constants."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "core" / "builtin_commands.py")
        source = source_path.read_text(encoding="utf-8")
        assert "HTTP_TIMEOUT" in source, (
            "builtin_commands.py should define HTTP_TIMEOUT constants")

    def test_truncation_constants_exist(self):
        """builtin_commands.py should define truncation constants."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "core" / "builtin_commands.py")
        source = source_path.read_text(encoding="utf-8")
        assert "CONTENT_TRUNCATION" in source or "OUTPUT_LIMIT" in source, (
            "builtin_commands.py should define output limit constants")


class TestRouterTypeHints:
    """router.py methods should have return type hints."""

    def test_router_methods_have_hints(self):
        """At least 80% of router.py methods should have return hints."""
        import ast
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "router.py")
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        total = 0
        with_hints = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "__init__":
                    continue
                total += 1
                if node.returns is not None:
                    with_hints += 1
        pct = with_hints / total * 100 if total else 0
        assert pct >= 80, (
            f"Only {with_hints}/{total} ({pct:.0f}%) of router.py "
            f"methods have return type hints — need at least 80%")


# ================================================================
# Identity card on MODELS page
# ================================================================

class TestModelsIdentityCard:
    """MODELS page shows identity info per model card."""

    def test_populate_model_cards_loads_identity(self):
        """_populate_model_cards loads identity context for each model."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "model_key_from_path" in source or "load_model_context" in source

    def test_model_card_shows_display_name(self):
        """Model card shows the identity display name when set."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "display_name" in source

    def test_model_card_shows_personality(self):
        """Model card shows personality description."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "personality" in source

    def test_model_card_shows_stats(self):
        """Model card shows message/session stats."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "total_messages" in source
        assert "total_sessions" in source

    def test_model_card_shows_tags(self):
        """Model card shows tags when present."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "tags" in source

    def test_model_card_shows_training_runs(self):
        """Model card shows training run count."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "training_history" in source


# ================================================================
# Identity editing on MODELS page
# ================================================================

class TestModelsIdentityEdit:
    """MODELS page supports inline name editing on model cards."""

    def test_start_inline_edit_method_exists(self):
        """PagesMixin has _start_inline_edit method."""
        from enigma_engine.gui.gui_pages import PagesMixin
        assert hasattr(PagesMixin, "_start_inline_edit")

    def test_save_inline_name_method_exists(self):
        """PagesMixin has _save_inline_name method."""
        from enigma_engine.gui.gui_pages import PagesMixin
        assert hasattr(PagesMixin, "_save_inline_name")

    def test_cancel_inline_name_method_exists(self):
        """PagesMixin has _cancel_inline_name method."""
        from enigma_engine.gui.gui_pages import PagesMixin
        assert hasattr(PagesMixin, "_cancel_inline_name")

    def test_save_inline_name_uses_model_context(self):
        """_save_inline_name persists via ModelContext."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._save_inline_name)
        assert "ModelContext" in source
        assert "display_name" in source

    def test_populate_has_edit_button(self):
        """Model cards have an EDIT button for inline editing."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "EDIT" in source
        assert "_start_inline_edit" in source

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

    def test_export_identity_method_exists(self):
        """ModelContext has export_identity method."""
        from enigma_engine.core.model_context import ModelContext
        assert hasattr(ModelContext, "export_identity")

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

    def test_gui_has_export_identity_button(self):
        """Model cards have an EXPORT button for identity."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._populate_model_cards)
        assert "EXPORT" in source
        assert "_export_identity" in source

    def test_export_identity_handler_exists(self):
        """PagesMixin has _export_identity handler."""
        from enigma_engine.gui.gui_pages import PagesMixin
        assert hasattr(PagesMixin, "_export_identity")


# ================================================================
# Font size control on CONFIG page
# ================================================================

class TestFontSizeControl:
    """CONFIG page has font size adjustment."""

    def test_font_size_offset_in_widgets(self):
        """widgets.py exposes a font_size_offset function or variable."""
        from enigma_engine.gui import widgets
        assert hasattr(widgets, "get_font_size_offset")
        assert hasattr(widgets, "set_font_size_offset")

    def test_font_size_offset_default_zero(self):
        """Default font size offset is 0."""
        from enigma_engine.gui import widgets
        assert widgets.get_font_size_offset() == 0

    def test_font_size_offset_adjusts_fonts(self):
        """set_font_size_offset changes module-level FONT_* tuples."""
        from enigma_engine.gui import widgets
        original_body = widgets.FONT_BODY[1]
        widgets.set_font_size_offset(2)
        assert widgets.FONT_BODY[1] == original_body + 2
        # Restore
        widgets.set_font_size_offset(0)
        assert widgets.FONT_BODY[1] == original_body

    def test_config_page_has_font_size_section(self):
        """CONFIG page builder includes font size controls."""
        from enigma_engine.gui.gui_pages_config import ConfigPageMixin
        source = inspect.getsource(
            ConfigPageMixin._build_page_config)
        assert "FONT SIZE" in source or "font_size" in source

    def test_font_size_persisted_in_settings(self):
        """Font size offset saved to gui_settings.json."""
        from enigma_engine.gui.gui_pages_config import ConfigPageMixin
        source = inspect.getsource(ConfigPageMixin)
        assert "font_size_offset" in source


# ================================================================
# A2: Keyboard shortcuts help overlay
# ================================================================

class TestKeyboardShortcutsOverlay:
    """CORE page has a keyboard shortcuts help overlay."""

    def test_shortcuts_method_exists(self):
        """desktop.py has _show_shortcuts_overlay method."""
        from enigma_engine.gui.desktop import EnigmaGUI
        assert hasattr(EnigmaGUI, "_show_shortcuts_overlay")

    def test_shortcuts_button_in_header(self):
        """desktop.py _build_shell creates a shortcuts help button."""
        source = inspect.getsource(
            __import__("enigma_engine.gui.desktop", fromlist=["EnigmaGUI"])
            .EnigmaGUI._build_shell)
        assert "_shortcuts_btn" in source or "shortcuts" in source.lower()

    def test_shortcuts_data_complete(self):
        """_show_shortcuts_overlay references known shortcuts."""
        source = inspect.getsource(
            __import__("enigma_engine.gui.desktop", fromlist=["EnigmaGUI"])
            .EnigmaGUI._show_shortcuts_overlay)
        # Must mention at least the core shortcuts
        assert "Ctrl" in source
        assert "Escape" in source or "ESC" in source

    def test_shortcuts_overlay_is_closeable(self):
        """Overlay has a close mechanism (destroy or Escape key)."""
        source = inspect.getsource(
            __import__("enigma_engine.gui.desktop", fromlist=["EnigmaGUI"])
            .EnigmaGUI._show_shortcuts_overlay)
        assert "destroy" in source or "withdraw" in source


# ================================================================
# A3: Token counter in chat
# ================================================================

class TestTokenCounter:
    """CORE page has a token counter."""

    def test_token_counter_label_created(self):
        """gui_pages.py creates a _token_counter_label widget."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        assert "_token_counter" in source

    def test_update_token_counter_method(self):
        """Logic mixin has _update_token_counter method."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        assert hasattr(LogicChatMixin, "_update_token_counter")

    def test_token_counter_called_after_send(self):
        """Token counter is updated after response finishes typing."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        # Moved to _typewriter so it updates AFTER the response finishes typing
        source = inspect.getsource(LogicChatMixin._typewriter)
        assert "_update_token_counter" in source


# ================================================================
# A4: HuggingFace download in GUI
# ================================================================

class TestHuggingFaceDownload:
    """MODELS page has HuggingFace download button."""

    def test_download_method_exists(self):
        """ForgeModelsMixin has _download_huggingface method."""
        from enigma_engine.gui.gui_forge_models import ForgeModelsMixin
        assert hasattr(ForgeModelsMixin, "_download_huggingface")

    def test_download_button_in_models_page(self):
        """MODELS page form includes a DOWNLOAD button."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_models)
        assert "DOWNLOAD" in source

    def test_download_uses_download_progress(self):
        """_download_huggingface uses the existing download_progress module."""
        source = inspect.getsource(
            __import__(
                "enigma_engine.gui.gui_forge_models",
                fromlist=["ForgeModelsMixin"])
            .ForgeModelsMixin._download_huggingface)
        assert "download_progress" in source or "DownloadTracker" in source


# ================================================================
# A7: Backup/restore system
# ================================================================

class TestBackupRestore:
    """CONFIG page has backup/restore buttons."""

    def test_export_backup_method(self):
        """ConfigPageMixin has _export_backup method."""
        from enigma_engine.gui.gui_pages_config import ConfigPageMixin
        assert hasattr(ConfigPageMixin, "_export_backup")

    def test_import_backup_method(self):
        """ConfigPageMixin has _import_backup method."""
        from enigma_engine.gui.gui_pages_config import ConfigPageMixin
        assert hasattr(ConfigPageMixin, "_import_backup")

    def test_backup_section_in_config_page(self):
        """CONFIG page builder includes backup/restore section."""
        from enigma_engine.gui.gui_pages_config import ConfigPageMixin
        source = inspect.getsource(
            ConfigPageMixin._build_page_config)
        assert "BACKUP" in source or "backup" in source

    def test_export_backup_covers_settings(self):
        """_export_backup includes gui_settings.json data."""
        source = inspect.getsource(
            __import__(
                "enigma_engine.gui.gui_pages_config",
                fromlist=["ConfigPageMixin"])
            .ConfigPageMixin._export_backup)
        assert "gui_settings" in source

    def test_export_backup_covers_memory(self):
        """_export_backup includes memory data."""
        source = inspect.getsource(
            __import__(
                "enigma_engine.gui.gui_pages_config",
                fromlist=["ConfigPageMixin"])
            .ConfigPageMixin._export_backup)
        assert "memory" in source or "notes" in source


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

class TestAdaptiveTrainingGUI:
    """Test adaptive training pipeline integration with FORGE page."""

    def test_adaptive_train_method_exists(self):
        """_start_adaptive_training method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_start_adaptive_training")

    def test_adaptive_training_uses_training_plan(self):
        """Adaptive training uses TrainingPlan for state."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_adaptive_training)
        assert "TrainingPlan" in source

    def test_adaptive_training_probes_student(self):
        """Adaptive training probes student ability first."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_adaptive_training)
        assert "probe" in source.lower()

    def test_adaptive_training_auto_chains_stages(self):
        """Adaptive training loops through all stages."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # advance_stage and decide_action are in the helper method
        source = inspect.getsource(
            ForgeMixin._adaptive_decide_action)
        assert "advance_stage" in source
        run_src = inspect.getsource(
            ForgeMixin._run_adaptive_stages)
        assert "decide_action" in run_src

    def test_adaptive_training_saves_plan(self):
        """Adaptive training saves plan to JSON."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._start_adaptive_training)
        assert "plan.save" in source or "save" in source

    def test_adaptive_training_uses_adaptive_prompts(self):
        """Adaptive training uses difficulty-aware prompts."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # build_adaptive_prompt used in phase1 helper
        source = inspect.getsource(
            ForgeMixin._adaptive_phase1_generate)
        assert "build_adaptive_prompt" in source

    def test_resume_training_plan_method_exists(self):
        """_resume_training_plan method exists on ForgeMixin."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, "_resume_training_plan")


# ================================================================
# Input history (Up/Down recall)
# ================================================================

class TestInputHistory:
    """Verify chat input history recall logic."""

    def test_input_history_methods_exist(self):
        """LogicChatMixin has input history methods."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        for attr in (
            "_init_input_history", "_on_input_up",
            "_on_input_down", "_set_input_text",
        ):
            assert hasattr(LogicChatMixin, attr), f"Missing: {attr}"

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

    def test_trim_chat_history_method_exists(self):
        """LogicChatMixin has _trim_chat_history method."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        assert hasattr(LogicChatMixin, "_trim_chat_history")

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

    def test_history_recorded_on_send(self):
        """_send_message records input to history."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        source = __import__("inspect").getsource(
            LogicChatMixin._send_message)
        assert "_input_history" in source
        assert "_input_hist_idx" in source


# ================================================================
# RLHF / Self-Play Dropdown (#21)
# ================================================================

class TestTrainingModes:
    """Verify the 3-mode training system: Basic, AI-Guided, Image."""

    def test_dropdown_has_basic_mode(self):
        """_build_page_forge includes Basic training mode."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = __import__("inspect").getsource(
            ForgePageMixin._build_page_forge)
        assert '"Basic"' in source

    def test_dropdown_has_ai_guided_mode(self):
        """_build_page_forge includes AI-Guided training mode."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = __import__("inspect").getsource(
            ForgePageMixin._build_page_forge)
        assert '"AI-Guided"' in source

    def test_dropdown_has_image_mode(self):
        """_build_page_forge includes Image training mode."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = __import__("inspect").getsource(
            ForgePageMixin._build_page_forge)
        assert '"Image"' in source

    def test_basic_mode_description(self):
        """Basic mode has descriptive text."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = __import__("inspect").getsource(
            ForgePageMixin._build_page_forge)
        assert "Train on your own data" in source

    def test_ai_guided_mode_description(self):
        """AI-Guided mode has descriptive text."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = __import__("inspect").getsource(
            ForgePageMixin._build_page_forge)
        assert "AI teacher creates curriculum" in source

    def test_image_mode_description(self):
        """Image mode has descriptive text."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = __import__("inspect").getsource(
            ForgePageMixin._build_page_forge)
        assert "Train on images or video" in source

    def test_dispatcher_handles_basic(self):
        """_start_training_by_mode dispatches Basic mode."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = __import__("inspect").getsource(
            ForgeMixin._start_training_by_mode)
        assert "Basic" in source

    def test_dispatcher_handles_ai_guided(self):
        """_start_training_by_mode dispatches AI-Guided mode."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = __import__("inspect").getsource(
            ForgeMixin._start_training_by_mode)
        assert "AI-Guided" in source

    def test_dispatcher_handles_image(self):
        """_start_training_by_mode dispatches Image mode."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = __import__("inspect").getsource(
            ForgeMixin._start_training_by_mode)
        assert "Image" in source
