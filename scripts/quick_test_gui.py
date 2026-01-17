#!/usr/bin/env python3
"""
Quick Test Script - Tests ForgeAI GUI with a small/fast model.

This script:
1. Loads ForgeAI with the smallest available model for faster testing
2. Shows the GUI so you can test UI changes with actual AI
3. Skips slow initialization where possible

Usage:
    python scripts/quick_test_gui.py
    python scripts/quick_test_gui.py --no-model  # UI only, no AI loading
"""

import sys
import os
import argparse

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Quick test ForgeAI GUI")
    parser.add_argument('--no-model', action='store_true', 
                       help='Skip model loading (UI only)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to load')
    args = parser.parse_args()
    
    print("=" * 50)
    print("ForgeAI Quick Test")
    print("=" * 50)
    
    # Import PyQt
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
    except ImportError:
        print("ERROR: PyQt5 not installed. Run: pip install PyQt5")
        return 1
    
    # Create app
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Import ForgeAI GUI
    try:
        from forge_ai.gui.enhanced_window import EnhancedMainWindow
    except ImportError as e:
        print(f"ERROR: Could not import ForgeAI GUI: {e}")
        return 1
    
    print("\n[1/3] Creating main window...")
    
    # Create window
    window = EnhancedMainWindow()
    
    if args.no_model:
        print("[2/3] Skipping model loading (--no-model)")
        window.setWindowTitle("ForgeAI - No Model (UI Test)")
    else:
        print("[2/3] Model will load when window opens...")
        if args.model:
            window.current_model_name = args.model
    
    print("[3/3] Showing window...")
    window.show()
    
    print("\n" + "=" * 50)
    print("GUI is running! Close the window to exit.")
    print("=" * 50 + "\n")
    
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
