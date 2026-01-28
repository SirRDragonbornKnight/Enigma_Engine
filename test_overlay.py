#!/usr/bin/env python3
"""
Test script for the AI Overlay system.

This script tests the overlay functionality including:
- Window creation with transparency
- Different display modes (MINIMAL, COMPACT, FULL)
- Theme switching
- Position presets
- Opacity changes

Run with: python test_overlay.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QLabel
    from PyQt5.QtCore import Qt
    from forge_ai.gui.overlay import AIOverlay, OverlayMode, OverlayPosition, OVERLAY_THEMES
    
    HAS_PYQT = True
except ImportError as e:
    print(f"⚠ Warning: PyQt5 not available: {e}")
    print("Install with: pip install PyQt5")
    HAS_PYQT = False


def main():
    """Test the overlay system."""
    if not HAS_PYQT:
        print("Cannot run overlay test without PyQt5")
        return 1
    
    print("=" * 60)
    print("AI Overlay Test")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    
    # Create control panel
    control_panel = QWidget()
    control_panel.setWindowTitle("Overlay Control Panel")
    control_panel.resize(400, 500)
    layout = QVBoxLayout(control_panel)
    
    # Title
    title = QLabel("AI Overlay Test Controls")
    title.setStyleSheet("font-size: 16px; font-weight: bold;")
    layout.addWidget(title)
    
    # Create overlay
    print("\n1. Creating overlay...")
    overlay = AIOverlay()
    print("   ✓ Overlay created")
    
    # Test different modes
    print("\n2. Testing overlay modes...")
    
    def test_minimal():
        print("   → Setting MINIMAL mode")
        overlay.set_mode(OverlayMode.MINIMAL)
        overlay.show()
    
    def test_compact():
        print("   → Setting COMPACT mode")
        overlay.set_mode(OverlayMode.COMPACT)
        overlay.show()
    
    def test_full():
        print("   → Setting FULL mode")
        overlay.set_mode(OverlayMode.FULL)
        overlay.show()
    
    def test_hidden():
        print("   → Setting HIDDEN mode")
        overlay.set_mode(OverlayMode.HIDDEN)
    
    # Mode buttons
    layout.addWidget(QLabel("\nDisplay Modes:"))
    
    btn_minimal = QPushButton("Minimal Mode")
    btn_minimal.clicked.connect(test_minimal)
    layout.addWidget(btn_minimal)
    
    btn_compact = QPushButton("Compact Mode")
    btn_compact.clicked.connect(test_compact)
    layout.addWidget(btn_compact)
    
    btn_full = QPushButton("Full Mode")
    btn_full.clicked.connect(test_full)
    layout.addWidget(btn_full)
    
    btn_hidden = QPushButton("Hidden Mode")
    btn_hidden.clicked.connect(test_hidden)
    layout.addWidget(btn_hidden)
    
    # Test themes
    print("\n3. Testing overlay themes...")
    layout.addWidget(QLabel("\nThemes:"))
    
    for theme_name in OVERLAY_THEMES.keys():
        def make_theme_setter(name):
            def set_theme():
                print(f"   → Setting {name} theme")
                overlay.set_theme(name)
            return set_theme
        
        btn = QPushButton(f"{theme_name.title()} Theme")
        btn.clicked.connect(make_theme_setter(theme_name))
        layout.addWidget(btn)
    
    # Test positions
    print("\n4. Testing overlay positions...")
    layout.addWidget(QLabel("\nPositions:"))
    
    positions = [
        ("Top Left", OverlayPosition.TOP_LEFT),
        ("Top Right", OverlayPosition.TOP_RIGHT),
        ("Bottom Left", OverlayPosition.BOTTOM_LEFT),
        ("Bottom Right", OverlayPosition.BOTTOM_RIGHT),
        ("Center", OverlayPosition.CENTER),
    ]
    
    for label, pos in positions:
        def make_pos_setter(position):
            def set_pos():
                print(f"   → Setting position: {position.value}")
                overlay.set_position(position)
            return set_pos
        
        btn = QPushButton(label)
        btn.clicked.connect(make_pos_setter(pos))
        layout.addWidget(btn)
    
    # Test opacity
    print("\n5. Testing overlay opacity...")
    layout.addWidget(QLabel("\nOpacity:"))
    
    for opacity in [0.3, 0.5, 0.7, 0.9, 1.0]:
        def make_opacity_setter(value):
            def set_opacity():
                print(f"   → Setting opacity: {value}")
                overlay.set_opacity(value)
            return set_opacity
        
        btn = QPushButton(f"Opacity {int(opacity * 100)}%")
        btn.clicked.connect(make_opacity_setter(opacity))
        layout.addWidget(btn)
    
    # Close button
    layout.addWidget(QLabel("\n"))
    btn_close = QPushButton("Close Test")
    btn_close.setStyleSheet("""
        QPushButton {
            background-color: #dc2626;
            color: white;
            font-weight: bold;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #b91c1c;
        }
    """)
    btn_close.clicked.connect(app.quit)
    layout.addWidget(btn_close)
    
    # Show control panel
    control_panel.show()
    
    # Start in compact mode
    overlay.set_mode(OverlayMode.COMPACT)
    overlay.set_position(OverlayPosition.TOP_RIGHT)
    overlay.show()
    
    print("\n" + "=" * 60)
    print("Overlay test running!")
    print("Use the control panel to test different modes, themes,")
    print("positions, and opacity levels.")
    print("=" * 60)
    
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
