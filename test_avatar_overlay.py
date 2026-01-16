#!/usr/bin/env python3
"""
Quick test script for the avatar overlay window.

Run this to test:
- Avatar shows on desktop
- Drag to move works
- Scroll to resize works  
- Right-click to hide works
- Double-click to reset size works
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor

# Import the avatar system
from forge_ai.gui.tabs.avatar.avatar_display import AvatarOverlayWindow
from forge_ai.avatar.renderers.default_sprites import generate_sprite
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import QByteArray

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create the overlay
    overlay = AvatarOverlayWindow()
    
    # Generate a sprite
    svg_data = generate_sprite(
        "happy",  # Use happy expression
        "#6366f1",  # Primary - blue
        "#8b5cf6",  # Secondary - purple
        "#10b981"   # Accent - green
    )
    
    # Convert SVG to pixmap
    renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
    pixmap = QPixmap(280, 280)
    pixmap.fill(QColor(0, 0, 0, 0))  # Transparent
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    
    # Set the avatar
    overlay.set_avatar(pixmap)
    
    # Show it
    overlay.show()
    overlay.raise_()
    
    print("=" * 50)
    print("Avatar Overlay Test")
    print("=" * 50)
    print()
    print("Controls:")
    print("  - Drag anywhere to move the avatar")
    print("  - Scroll wheel to resize")
    print("  - Right-click to hide")
    print("  - Double-click to reset size")
    print()
    print("Close the avatar to exit the test.")
    print("=" * 50)
    
    # Connect close signal to quit
    overlay.closed.connect(app.quit)
    
    # Run the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
