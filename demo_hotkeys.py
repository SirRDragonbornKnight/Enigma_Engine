#!/usr/bin/env python3
"""
Demo script for the hotkey system.

This demonstrates how to use the global hotkey system programmatically.
"""

import time
import sys


def demo_basic_usage():
    """Demonstrate basic hotkey registration."""
    print("\n" + "=" * 60)
    print("DEMO: Basic Hotkey Registration")
    print("=" * 60)
    
    try:
        # Import manager
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'hotkey_manager',
            'forge_ai/core/hotkey_manager.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Create manager
        manager = mod.HotkeyManager()
        print("✓ HotkeyManager created")
        
        # Define a callback
        callback_called = []
        
        def my_callback():
            callback_called.append(time.time())
            print(f"  → Callback triggered! (count: {len(callback_called)})")
        
        # Register a hotkey
        success = manager.register("Ctrl+Shift+T", my_callback, "demo_hotkey")
        
        if success:
            print(f"✓ Hotkey registered: Ctrl+Shift+T")
            print(f"✓ Total hotkeys: {len(manager.list_registered())}")
            
            # List registered hotkeys
            print("\nRegistered hotkeys:")
            for hotkey in manager.list_registered():
                print(f"  - {hotkey['name']}: {hotkey['hotkey']} (enabled: {hotkey['enabled']})")
        else:
            print("✗ Hotkey registration failed (backend not available)")
        
        # Cleanup
        manager.unregister("demo_hotkey")
        print("\n✓ Hotkey unregistered")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_default_hotkeys():
    """Demonstrate default hotkey definitions."""
    print("\n" + "=" * 60)
    print("DEMO: Default Hotkeys")
    print("=" * 60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'hotkey_manager',
            'forge_ai/core/hotkey_manager.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        print("\nDefault hotkey bindings:")
        for name, hotkey in mod.DEFAULT_HOTKEYS.items():
            print(f"  {name:20s} → {hotkey}")
        
        print("\n✓ All default hotkeys defined")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")


def demo_actions():
    """Demonstrate hotkey actions."""
    print("\n" + "=" * 60)
    print("DEMO: Hotkey Actions")
    print("=" * 60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'hotkey_actions',
            'forge_ai/core/hotkey_actions.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Create actions instance
        actions = mod.HotkeyActions()
        print("✓ HotkeyActions created")
        
        # Test game mode toggle
        print("\nTesting game mode toggle:")
        initial = actions._game_mode_active
        print(f"  Initial state: {initial}")
        
        actions.toggle_game_mode()
        print(f"  After toggle: {actions._game_mode_active}")
        
        actions.toggle_game_mode()
        print(f"  After 2nd toggle: {actions._game_mode_active}")
        
        print("\n✓ Game mode toggle working")
        
        # Test other actions (without GUI)
        print("\nAvailable actions:")
        action_methods = [
            'summon_overlay',
            'dismiss_overlay',
            'push_to_talk_start',
            'push_to_talk_stop',
            'quick_command',
            'screenshot_to_ai',
            'toggle_game_mode'
        ]
        
        for method in action_methods:
            if hasattr(actions, method):
                print(f"  ✓ {method}")
            else:
                print(f"  ✗ {method} (missing)")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_config_integration():
    """Demonstrate config integration."""
    print("\n" + "=" * 60)
    print("DEMO: Configuration Integration")
    print("=" * 60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'defaults',
            'forge_ai/config/defaults.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Check enable flag
        enabled = mod.CONFIG.get('enable_hotkeys', False)
        print(f"\nGlobal hotkeys enabled: {enabled}")
        
        # Check hotkey config
        hotkeys = mod.CONFIG.get('hotkeys', {})
        print(f"\nConfigured hotkeys:")
        for name, key in hotkeys.items():
            print(f"  {name:20s} → {key}")
        
        print("\n✓ Configuration loaded successfully")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_platform_detection():
    """Demonstrate platform detection."""
    print("\n" + "=" * 60)
    print("DEMO: Platform Detection")
    print("=" * 60)
    
    print(f"\nCurrent platform: {sys.platform}")
    
    if sys.platform == 'win32':
        print("  → Using Windows backend (ctypes/win32api)")
    elif sys.platform == 'darwin':
        print("  → Using macOS backend (Quartz/keyboard)")
    else:
        print("  → Using Linux backend (Xlib/keyboard)")
    
    print("\n✓ Platform detected")


def main():
    """Run all demos."""
    print("=" * 60)
    print("HOTKEY SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    demo_platform_detection()
    demo_default_hotkeys()
    demo_config_integration()
    demo_basic_usage()
    demo_actions()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nTo use hotkeys in ForgeAI:")
    print("  1. Launch ForgeAI GUI: python run.py --gui")
    print("  2. Go to Settings tab")
    print("  3. Scroll to Global Hotkeys section")
    print("  4. Enable/disable or rebind hotkeys")
    print("\n")


if __name__ == "__main__":
    main()
