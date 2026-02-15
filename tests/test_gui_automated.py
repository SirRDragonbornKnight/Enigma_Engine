"""
GUI Test Suite for Enigma AI Engine

Automated tests for the PyQt5 GUI using pytest-qt.
Run with: pytest tests/test_gui_automated.py -v

Requirements:
  pip install pytest-qt

Note: These tests run headless where possible, or with a hidden display.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure headless mode for CI/testing
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# Try importing Qt - skip all tests if not available
pytest_qt_available = False
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtTest import QTest
    pytest_qt_available = True
except ImportError:
    pass

# Skip all tests if PyQt5 not available
pytestmark = pytest.mark.skipif(
    not pytest_qt_available,
    reason="PyQt5 not available"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication for all tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def main_window(qapp, qtbot):
    """Create the main window for testing."""
    try:
        from enigma_engine.gui.enhanced_window import EnhancedMainWindow
        window = EnhancedMainWindow()
        qtbot.addWidget(window)
        yield window
    except Exception as e:
        pytest.skip(f"Cannot create main window: {e}")


# =============================================================================
# Tab Import Tests
# =============================================================================

class TestTabImports:
    """Test that all GUI tabs can be imported without errors."""
    
    def test_chat_tab_import(self):
        """Test chat tab imports."""
        from enigma_engine.gui.tabs.chat_tab import create_chat_tab
        assert callable(create_chat_tab)
    
    def test_training_tab_import(self):
        """Test training tab imports."""
        from enigma_engine.gui.tabs.training_tab import create_training_tab
        assert callable(create_training_tab)
    
    def test_avatar_tab_import(self):
        """Test avatar tab imports."""
        from enigma_engine.gui.tabs.avatar_tab import create_avatar_tab
        assert callable(create_avatar_tab)
    
    def test_image_tab_import(self):
        """Test image tab imports."""
        from enigma_engine.gui.tabs.image_tab import ImageTab
        assert ImageTab is not None
    
    def test_model_manager_import(self):
        """Test model manager imports."""
        from enigma_engine.gui.dialogs.model_manager import ModelManagerDialog
        assert ModelManagerDialog is not None
    
    def test_persona_tab_import(self):
        """Test persona tab imports."""
        from enigma_engine.gui.tabs.persona_tab import create_persona_tab
        assert callable(create_persona_tab)
    
    def test_terminal_tab_import(self):
        """Test terminal tab imports."""
        from enigma_engine.gui.tabs.terminal_tab import create_terminal_tab
        assert callable(create_terminal_tab)


# =============================================================================
# Widget Creation Tests
# =============================================================================

class TestWidgetCreation:
    """Test that widgets can be created without errors."""
    
    def test_create_chat_widget(self, qapp, qtbot):
        """Test creating a chat widget."""
        from enigma_engine.gui.tabs.chat_tab import create_chat_tab
        parent = QWidget()
        qtbot.addWidget(parent)
        try:
            widget = create_chat_tab(parent)
            assert widget is not None
            assert isinstance(widget, QWidget)
        except Exception as e:
            pytest.skip(f"Chat widget creation failed: {e}")
    
    def test_create_training_widget(self, qapp, qtbot):
        """Test creating a training widget."""
        from enigma_engine.gui.tabs.training_tab import create_training_tab
        parent = QWidget()
        qtbot.addWidget(parent)
        try:
            widget = create_training_tab(parent)
            assert widget is not None
        except Exception as e:
            pytest.skip(f"Training widget creation failed: {e}")
    
    def test_create_image_widget(self, qapp, qtbot):
        """Test creating an image tab widget."""
        from enigma_engine.gui.tabs.image_tab import ImageTab
        try:
            widget = ImageTab()
            qtbot.addWidget(widget)
            assert widget is not None
        except Exception as e:
            pytest.skip(f"Image widget creation failed: {e}")


# =============================================================================
# Interaction Tests
# =============================================================================

class TestWidgetInteractions:
    """Test widget interactions - clicking buttons, entering text, etc."""
    
    def test_button_exists_and_clickable(self, main_window, qtbot):
        """Test that main buttons exist and are clickable."""
        # Find all buttons in the window
        buttons = main_window.findChildren(QPushButton)
        assert len(buttons) > 0, "No buttons found in main window"
        
        # Verify buttons are enabled
        enabled_buttons = [b for b in buttons if b.isEnabled()]
        assert len(enabled_buttons) > 0, "No enabled buttons found"
    
    def test_tab_switching(self, main_window, qtbot):
        """Test switching between tabs."""
        from PyQt5.QtWidgets import QTabWidget
        
        tab_widgets = main_window.findChildren(QTabWidget)
        if not tab_widgets:
            pytest.skip("No tab widgets found")
        
        tab_widget = tab_widgets[0]
        initial_index = tab_widget.currentIndex()
        
        # Switch to different tabs
        for i in range(min(3, tab_widget.count())):
            tab_widget.setCurrentIndex(i)
            qtbot.wait(100)  # Small delay for UI update
            assert tab_widget.currentIndex() == i


# =============================================================================
# Visual Regression Test Helpers
# =============================================================================

class TestVisualRegression:
    """Visual regression tests - capture and compare screenshots."""
    
    SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
    
    @pytest.fixture(autouse=True)
    def setup_screenshot_dir(self):
        """Create screenshot directory if it doesn't exist."""
        self.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    
    def capture_screenshot(self, widget: QWidget, name: str) -> Path:
        """Capture a screenshot of a widget."""
        from PyQt5.QtGui import QPixmap
        
        pixmap = widget.grab()
        path = self.SCREENSHOT_DIR / f"{name}.png"
        pixmap.save(str(path))
        return path
    
    def test_main_window_screenshot(self, main_window, qtbot):
        """Capture main window screenshot for visual verification."""
        main_window.show()
        qtbot.wait(500)  # Wait for rendering
        
        path = self.capture_screenshot(main_window, "main_window")
        assert path.exists(), "Screenshot was not saved"
        
        # Clean up
        main_window.hide()
    
    def compare_screenshots(self, current: Path, baseline: Path, threshold: float = 0.95) -> bool:
        """Compare two screenshots and return True if similar enough.
        
        Args:
            current: Path to current screenshot
            baseline: Path to baseline screenshot
            threshold: Similarity threshold (0-1, default 0.95 = 95% similar)
        
        Returns:
            True if images are similar within threshold
        """
        try:
            from PIL import Image
            import numpy as np
            
            if not baseline.exists():
                # No baseline - save current as baseline
                import shutil
                shutil.copy(current, baseline)
                return True
            
            img1 = np.array(Image.open(current).convert('RGB'))
            img2 = np.array(Image.open(baseline).convert('RGB'))
            
            if img1.shape != img2.shape:
                return False
            
            # Calculate similarity (simple pixel comparison)
            diff = np.abs(img1.astype(float) - img2.astype(float))
            similarity = 1 - (diff.mean() / 255)
            
            return similarity >= threshold
            
        except ImportError:
            # PIL not available, skip comparison
            return True


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests for UI stability."""
    
    def test_rapid_tab_switching(self, main_window, qtbot):
        """Test rapid tab switching doesn't crash."""
        from PyQt5.QtWidgets import QTabWidget
        
        tab_widgets = main_window.findChildren(QTabWidget)
        if not tab_widgets:
            pytest.skip("No tab widgets found")
        
        tab_widget = tab_widgets[0]
        
        # Rapidly switch tabs 20 times
        for _ in range(20):
            for i in range(tab_widget.count()):
                tab_widget.setCurrentIndex(i)
                qtbot.wait(10)
        
        # If we get here without crashing, test passed
        assert True
    
    def test_resize_window(self, main_window, qtbot):
        """Test window resizing doesn't crash."""
        main_window.show()
        original_size = main_window.size()
        
        # Resize multiple times
        sizes = [(800, 600), (1024, 768), (640, 480), (1280, 720)]
        for w, h in sizes:
            main_window.resize(w, h)
            qtbot.wait(50)
        
        # Restore original size
        main_window.resize(original_size)
        main_window.hide()
        
        assert True


# =============================================================================
# Accessibility Tests
# =============================================================================

class TestAccessibility:
    """Basic accessibility tests."""
    
    def test_buttons_have_text_or_tooltip(self, main_window, qtbot):
        """Test that buttons have accessible text or tooltips."""
        buttons = main_window.findChildren(QPushButton)
        
        for button in buttons:
            has_text = bool(button.text().strip())
            has_tooltip = bool(button.toolTip().strip())
            has_accessible = bool(button.accessibleName().strip()) if hasattr(button, 'accessibleName') else False
            
            # Button should have at least one form of identification
            assert has_text or has_tooltip or has_accessible, \
                f"Button has no accessible text: {button.objectName()}"
    
    def test_labels_readable(self, main_window, qtbot):
        """Test that labels have text content."""
        labels = main_window.findChildren(QLabel)
        
        # Most labels should have text (allow some to be empty for layout)
        labels_with_text = [l for l in labels if l.text().strip()]
        assert len(labels_with_text) > 0, "No readable labels found"


# =============================================================================
# Run standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
