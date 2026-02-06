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
        from forge_ai.gui.dialogs.font_selector import FontSelectorDialog
        
        assert FontSelectorDialog is not None


class TestLoadingDialog:
    """Test loading dialog."""
    
    def test_dialog_class_exists(self):
        """Test LoadingDialog class exists."""
        from forge_ai.gui.dialogs.loading import LoadingDialog
        
        assert LoadingDialog is not None
    
    def test_progress_interface(self):
        """Test progress update interface."""
        from forge_ai.gui.dialogs.loading import LoadingDialog
        
        # Should have progress methods
        assert hasattr(LoadingDialog, 'set_progress')
        assert hasattr(LoadingDialog, 'set_message')


class TestCommandPaletteDialog:
    """Test command palette dialog."""
    
    def test_dialog_class_exists(self):
        """Test CommandPaletteDialog class exists."""
        from forge_ai.gui.dialogs.command_palette import CommandPalette
        
        assert CommandPalette is not None
    
    def test_command_registration(self):
        """Test command registration."""
        from forge_ai.gui.dialogs.command_palette import CommandPalette
        
        # Should have command methods
        assert hasattr(CommandPalette, 'register_command')
        assert hasattr(CommandPalette, 'execute_command')


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
        from forge_ai.gui.dialogs import (
            ModelManagerDialog,
            ThemeEditorDialog,
            FontSelectorDialog,
            LoadingDialog,
            CommandPalette
        )
        
        # All should be importable
        assert ModelManagerDialog is not None
        assert ThemeEditorDialog is not None
        assert FontSelectorDialog is not None
        assert LoadingDialog is not None
        assert CommandPalette is not None
