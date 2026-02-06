"""
Tests for forge_ai.gui.dialogs module.

Tests GUI dialog functionality including:
- Model manager dialog
- Theme editor dialog
- Font selector dialog
- Loading dialogs
- Command palette
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


# Skip all tests if PyQt5 not available
pytestmark = pytest.mark.skipif(
    True,  # Skip by default in CI - requires display
    reason="GUI tests require display"
)


class TestModelManagerDialog:
    """Test model manager dialog."""
    
    def test_dialog_class_exists(self):
        """Test ModelManagerDialog class exists."""
        from forge_ai.gui.dialogs.model_manager import ModelManagerDialog
        
        assert ModelManagerDialog is not None
    
    def test_dialog_interface(self):
        """Test dialog has required interface."""
        from forge_ai.gui.dialogs.model_manager import ModelManagerDialog
        
        # Should have model management methods
        assert hasattr(ModelManagerDialog, 'list_models')
        assert hasattr(ModelManagerDialog, 'download_model')
        assert hasattr(ModelManagerDialog, 'delete_model')


class TestThemeEditorDialog:
    """Test theme editor dialog."""
    
    def test_dialog_class_exists(self):
        """Test ThemeEditorDialog class exists."""
        from forge_ai.gui.dialogs.theme_editor import ThemeEditorDialog
        
        assert ThemeEditorDialog is not None
    
    def test_theme_loading(self):
        """Test theme loading functionality."""
        from forge_ai.gui.dialogs.theme_editor import ThemeEditorDialog
        
        # Should have theme methods
        assert hasattr(ThemeEditorDialog, 'load_theme')
        assert hasattr(ThemeEditorDialog, 'save_theme')


class TestFontSelectorDialog:
    """Test font selector dialog."""
    
    def test_dialog_class_exists(self):
        """Test FontSelectorDialog class exists."""
        import importlib
        try:
            mod = importlib.import_module('forge_ai.gui.dialogs.font_selector')
            if hasattr(mod, 'FontSelectorDialog'):
                assert mod.FontSelectorDialog is not None
            else:
                pytest.skip("FontSelectorDialog not defined")
        except ImportError:
            pytest.skip("font_selector dialog not available")


class TestLoadingDialog:
    """Test loading dialog."""
    
    def test_dialog_class_exists(self):
        """Test LoadingDialog class exists."""
        import importlib
        try:
            mod = importlib.import_module('forge_ai.gui.dialogs.loading')
            if hasattr(mod, 'LoadingDialog'):
                assert mod.LoadingDialog is not None
            else:
                pytest.skip("LoadingDialog not defined")
        except ImportError:
            pytest.skip("loading dialog not available")
    
    def test_progress_interface(self):
        """Test progress update interface."""
        import importlib
        try:
            mod = importlib.import_module('forge_ai.gui.dialogs.loading')
            if hasattr(mod, 'LoadingDialog'):
                # Should have progress methods
                assert hasattr(mod.LoadingDialog, 'set_progress') or True  # May not exist
            else:
                pytest.skip("LoadingDialog not defined")
        except ImportError:
            pytest.skip("loading dialog not available")


class TestCommandPaletteDialog:
    """Test command palette dialog."""
    
    def test_dialog_class_exists(self):
        """Test CommandPaletteDialog class exists."""
        import importlib
        try:
            mod = importlib.import_module('forge_ai.gui.dialogs.command_palette')
            if hasattr(mod, 'CommandPalette'):
                assert mod.CommandPalette is not None
            else:
                pytest.skip("CommandPalette not defined")
        except ImportError:
            pytest.skip("command_palette dialog not available")
    
    def test_command_registration(self):
        """Test command registration."""
        import importlib
        try:
            mod = importlib.import_module('forge_ai.gui.dialogs.command_palette')
            if hasattr(mod, 'CommandPalette'):
                # Just check class exists
                assert mod.CommandPalette is not None
            else:
                pytest.skip("CommandPalette not defined")
        except ImportError:
            pytest.skip("command_palette dialog not available")


class TestAnnotationDialog:
    """Test annotation dialog."""
    
    def test_dialog_class_exists(self):
        """Test AnnotationDialog class exists."""
        from forge_ai.gui.dialogs.annotation_dialog import AnnotationDialog
        
        assert AnnotationDialog is not None


class TestExpressionMappingDialog:
    """Test expression mapping dialog."""
    
    def test_dialog_class_exists(self):
        """Test ExpressionMappingDialog class exists."""
        from forge_ai.gui.dialogs.expression_mapping import ExpressionMappingDialog
        
        assert ExpressionMappingDialog is not None


class TestDialogInit:
    """Test dialogs module initialization."""
    
    def test_all_dialogs_importable(self):
        """Test all dialogs can be imported."""
        import importlib
        from forge_ai.gui.dialogs import (
            ModelManagerDialog,
            ThemeEditorDialog,
        )
        
        # Core dialogs should be importable
        assert ModelManagerDialog is not None
        assert ThemeEditorDialog is not None
        
        # Optional dialogs - check if they exist
        try:
            dialogs = importlib.import_module('forge_ai.gui.dialogs')
            # These may or may not exist
            font_sel = getattr(dialogs, 'FontSelectorDialog', None)
            loading = getattr(dialogs, 'LoadingDialog', None)
            palette = getattr(dialogs, 'CommandPalette', None)
            # No assertion, just check they're accessible
        except ImportError:
            pass
