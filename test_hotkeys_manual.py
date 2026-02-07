#!/usr/bin/env python3
"""
Manual test script for the hotkey system.

This script tests the hotkey system without requiring PyQt5 or other heavy dependencies.
Run this to verify the core hotkey functionality works.

Usage:
    python test_hotkeys_manual.py
"""

import sys
import time


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import importlib.util
        
        # Test hotkey manager
        spec = importlib.util.spec_from_file_location(
            'hotkey_manager',
            'enigma_engine/core/hotkey_manager.py'
        )
        hotkey_manager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hotkey_manager)
        print("  ✓ hotkey_manager imported")
        
        # Test hotkey actions
        spec = importlib.util.spec_from_file_location(
            'hotkey_actions',
            'enigma_engine/core/hotkey_actions.py'
        )
        hotkey_actions = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hotkey_actions)
        print("  ✓ hotkey_actions imported")
        
        # Test backends
        if sys.platform == 'win32':
            spec = importlib.util.spec_from_file_location(
                'windows',
                'enigma_engine/core/hotkeys/windows.py'
            )
            windows = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(windows)
            print("  ✓ Windows backend imported")
        elif sys.platform == 'darwin':
            spec = importlib.util.spec_from_file_location(
                'macos',
                'enigma_engine/core/hotkeys/macos.py'
            )
            macos = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(macos)
            print("  ✓ macOS backend imported")
        else:
            spec = importlib.util.spec_from_file_location(
                'linux',
                'enigma_engine/core/hotkeys/linux.py'
            )
            linux = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(linux)
            print("  ✓ Linux backend imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_default_hotkeys():
    """Test default hotkey definitions."""
    print("\nTesting default hotkeys...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'hotkey_manager',
            'enigma_engine/core/hotkey_manager.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        expected_keys = [
            'summon_ai', 'dismiss_ai', 'push_to_talk',
            'toggle_game_mode', 'quick_command', 'screenshot_to_ai'
        ]
        
        for key in expected_keys:
            if key in mod.DEFAULT_HOTKEYS:
                print(f"  ✓ {key}: {mod.DEFAULT_HOTKEYS[key]}")
            else:
                print(f"  ✗ {key}: MISSING")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_manager_creation():
    """Test creating a HotkeyManager instance."""
    print("\nTesting HotkeyManager creation...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'hotkey_manager',
            'enigma_engine/core/hotkey_manager.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        manager = mod.HotkeyManager()
        print(f"  ✓ Manager created")
        print(f"  ✓ Running: {manager.is_running()}")
        print(f"  ✓ Registered hotkeys: {len(manager.list_registered())}")
        
        return True
    except Exception as e:
        print(f"  ✗ Creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_actions_creation():
    """Test creating a HotkeyActions instance."""
    print("\nTesting HotkeyActions creation...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'hotkey_actions',
            'enigma_engine/core/hotkey_actions.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        actions = mod.HotkeyActions()
        print(f"  ✓ Actions created")
        
        # Test toggle game mode
        initial = actions._game_mode_active
        actions.toggle_game_mode()
        after = actions._game_mode_active
        
        if initial != after:
            print(f"  ✓ Game mode toggle works")
        else:
            print(f"  ✗ Game mode toggle failed")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_integration():
    """Test config integration."""
    print("\nTesting config integration...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'defaults',
            'enigma_engine/config/defaults.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        if 'enable_hotkeys' in mod.CONFIG:
            print(f"  ✓ enable_hotkeys: {mod.CONFIG['enable_hotkeys']}")
        else:
            print(f"  ✗ enable_hotkeys: MISSING")
            return False
        
        if 'hotkeys' in mod.CONFIG:
            print(f"  ✓ hotkeys config exists")
            hotkeys = mod.CONFIG['hotkeys']
            for name, key in hotkeys.items():
                print(f"    - {name}: {key}")
        else:
            print(f"  ✗ hotkeys config: MISSING")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HOTKEY SYSTEM MANUAL TEST")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Default Hotkeys", test_default_hotkeys()))
    results.append(("Manager Creation", test_manager_creation()))
    results.append(("Actions Creation", test_actions_creation()))
    results.append(("Config Integration", test_config_integration()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
