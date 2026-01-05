"""
Tests for local vs cloud module distinction.

Tests the is_cloud_service field, local_only mode, and helper functions.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma.modules.manager import ModuleManager, ModuleInfo, ModuleCategory
from enigma.modules import registry


def test_cloud_modules_marked():
    """All *_api modules should have is_cloud_service=True."""
    print("Testing cloud modules are marked correctly...")
    cloud_module_ids = [
        'image_gen_api',
        'code_gen_api',
        'video_gen_api',
        'audio_gen_api',
        'embedding_api',
    ]
    
    for module_id in cloud_module_ids:
        module_class = registry.get_module(module_id)
        assert module_class is not None, f"Module {module_id} not found"
        
        info = module_class.get_info()
        assert info.is_cloud_service is True, \
            f"Module {module_id} should be marked as cloud service"
        print(f"  [OK] {module_id} correctly marked as cloud service")
    
    print("  All cloud modules marked correctly!\n")


def test_local_modules_marked():
    """All local modules should have is_cloud_service=False (default)."""
    print("Testing local modules are marked correctly...")
    local_module_ids = [
        'model',
        'tokenizer',
        'training',
        'inference',
        'memory',
        'image_gen_local',
        'code_gen_local',
        'video_gen_local',
        'audio_gen_local',
        'embedding_local',
        'voice_input',
        'voice_output',
        'vision',
        'avatar',
        'web_tools',
        'file_tools',
        'api_server',
        'network',
        'gui',
    ]
    
    for module_id in local_module_ids:
        module_class = registry.get_module(module_id)
        assert module_class is not None, f"Module {module_id} not found"
        
        info = module_class.get_info()
        assert info.is_cloud_service is False, \
            f"Module {module_id} should be marked as local (not cloud)"
    
    print(f"  [OK] All {len(local_module_ids)} local modules marked correctly!\n")


def test_local_only_default_true():
    """ModuleManager should default to local_only=True."""
    print("Testing local_only mode defaults to True...")
    try:
        manager = ModuleManager()
        assert manager.local_only is True, \
            "ModuleManager should default to local_only=True"
        print("  [OK] local_only defaults to True!\n")
    except ImportError as e:
        print(f"  [SKIP] Skipped (missing dependency: {e})\n")


def test_local_only_prevents_cloud_loading():
    """local_only=True should prevent loading cloud modules."""
    print("Testing local_only=True prevents cloud module loading...")
    try:
        manager = ModuleManager(local_only=True)
        
        # Register modules
        registry.register_all(manager)
        
        # Try to load a cloud module
        cloud_modules = ['image_gen_api', 'code_gen_api', 'embedding_api']
        
        for module_id in cloud_modules:
            can_load, reason = manager.can_load(module_id)
            assert can_load is False, \
                f"Should not be able to load {module_id} in local_only mode"
            assert "cloud" in reason.lower() or "external" in reason.lower(), \
                f"Reason should mention cloud/external services: {reason}"
            print(f"  [OK] {module_id} blocked in local_only mode")
        
        print("  Cloud modules correctly blocked!\n")
    except ImportError as e:
        print(f"  [SKIP] Skipped (missing dependency: {e})\n")


def test_local_only_false_allows_cloud():
    """local_only=False should allow loading cloud modules."""
    print("Testing local_only=False allows cloud modules...")
    try:
        manager = ModuleManager(local_only=False)
        
        # Register modules
        registry.register_all(manager)
        
        # Check that cloud modules can be loaded (hardware permitting)
        cloud_modules = ['code_gen_api', 'embedding_api']
        
        for module_id in cloud_modules:
            can_load, reason = manager.can_load(module_id)
            # Should pass cloud check (may fail on other checks like dependencies)
            if not can_load:
                assert "cloud" not in reason.lower() and "external" not in reason.lower(), \
                    f"Should not fail due to cloud restriction: {reason}"
            print(f"  [OK] {module_id} passes cloud check")
        
        print("  Cloud modules allowed when local_only=False!\n")
    except ImportError as e:
        print(f"  [SKIP] Skipped (missing dependency: {e})\n")


def test_list_local_modules():
    """list_local_modules() should return only local modules."""
    print("Testing list_local_modules() function...")
    local_modules = registry.list_local_modules()
    
    assert len(local_modules) > 0, "Should have local modules"
    
    for module_info in local_modules:
        assert module_info.is_cloud_service is False, \
            f"list_local_modules() returned cloud module: {module_info.id}"
    
    print(f"  [OK] Found {len(local_modules)} local modules")
    print("  All are correctly marked as local!\n")


def test_list_cloud_modules():
    """list_cloud_modules() should return only cloud modules."""
    print("Testing list_cloud_modules() function...")
    cloud_modules = registry.list_cloud_modules()
    
    assert len(cloud_modules) > 0, "Should have cloud modules"
    
    for module_info in cloud_modules:
        assert module_info.is_cloud_service is True, \
            f"list_cloud_modules() returned local module: {module_info.id}"
    
    print(f"  [OK] Found {len(cloud_modules)} cloud modules")
    print("  All are correctly marked as cloud!\n")


def test_all_modules_categorized():
    """All modules should be in either local or cloud list."""
    print("Testing all modules are categorized...")
    all_modules = registry.list_modules()
    local_modules = registry.list_local_modules()
    cloud_modules = registry.list_cloud_modules()
    
    assert len(all_modules) == len(local_modules) + len(cloud_modules), \
        "All modules should be categorized as either local or cloud"
    
    print(f"  [OK] Total: {len(all_modules)} modules")
    print(f"  [OK] Local: {len(local_modules)} modules")
    print(f"  [OK] Cloud: {len(cloud_modules)} modules")
    print("  All modules properly categorized!\n")


def test_module_info_field():
    """Test the is_cloud_service field in ModuleInfo."""
    print("Testing ModuleInfo.is_cloud_service field...")
    
    # Test default
    info = ModuleInfo(
        id="test",
        name="Test",
        description="Test module",
        category=ModuleCategory.CORE,
    )
    
    assert hasattr(info, 'is_cloud_service'), \
        "ModuleInfo should have is_cloud_service field"
    assert info.is_cloud_service is False, \
        "is_cloud_service should default to False"
    print("  [OK] Field exists and defaults to False")
    
    # Test can be set to True
    info_cloud = ModuleInfo(
        id="test",
        name="Test",
        description="Test module",
        category=ModuleCategory.CORE,
        is_cloud_service=True,
    )
    
    assert info_cloud.is_cloud_service is True, \
        "is_cloud_service should be settable to True"
    print("  [OK] Field can be set to True\n")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running Local vs Cloud Module Tests")
    print("="*60 + "\n")
    
    tests = [
        test_cloud_modules_marked,
        test_local_modules_marked,
        test_local_only_default_true,
        test_local_only_prevents_cloud_loading,
        test_local_only_false_allows_cloud,
        test_list_local_modules,
        test_list_cloud_modules,
        test_all_modules_categorized,
        test_module_info_field,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except ImportError:
            # Count as skipped, not failed
            skipped += 1
        except AssertionError as e:
            print(f"  [FAIL] FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] ERROR: {e}\n")
            failed += 1
    
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

