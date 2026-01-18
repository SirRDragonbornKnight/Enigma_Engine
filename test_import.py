#!/usr/bin/env python3
"""Quick test to verify avatar_display imports correctly."""
try:
    from forge_ai.gui.tabs.avatar.avatar_display import (
        create_avatar_subtab,
        _use_builtin_sprite,
        _show_default_preview,
        AvatarOverlayWindow
    )
    print("✓ All imports successful!")
    print("  - create_avatar_subtab")
    print("  - _use_builtin_sprite")
    print("  - _show_default_preview")
    print("  - AvatarOverlayWindow")
except Exception as e:
    print(f"✗ Import error: {e}")
