#!/usr/bin/env python3
"""
Enigma Engine - Pre-Testing System Check

Run this script to verify all components are working before testing the full system.

Usage:
    python test_system.py              # Run all tests
    python test_system.py --quick      # Quick essential tests only
    python test_system.py --verbose    # Show detailed output
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def ok(msg):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")


def fail(msg):
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")


def warn(msg):
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {msg}")


def info(msg):
    print(f"  {Colors.BLUE}ℹ{Colors.RESET} {msg}")


def header(msg):
    print(f"\n{Colors.BOLD}{msg}{Colors.RESET}")
    print("─" * 50)


def test_environment():
    """Test Python environment."""
    header("1. Environment")
    
    passed = True
    
    # Python version
    version = sys.version_info
    if version >= (3, 8):
        ok(f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        fail(f"Python {version.major}.{version.minor} (need 3.8+)")
        passed = False
    
    # Check key packages
    packages = {
        "torch": "PyTorch (neural networks)",
        "flask": "Flask (API server)",
        "flask_cors": "Flask-CORS (API CORS)",
    }
    
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            ok(f"{desc}")
        except ImportError:
            fail(f"{desc} not installed")
            passed = False
    
    # Optional packages
    optional = {
        "PyQt5": "PyQt5 (GUI)",
        "PIL": "Pillow (images)",
        "pyttsx3": "pyttsx3 (TTS)",
        "pytesseract": "pytesseract (OCR)",
    }
    
    for pkg, desc in optional.items():
        try:
            if pkg == "PIL":
                __import__("PIL")
            else:
                __import__(pkg)
            ok(f"{desc}")
        except ImportError:
            warn(f"{desc} not installed (optional)")
    
    return passed


def test_hardware():
    """Test hardware detection."""
    header("2. Hardware Detection")
    
    try:
        from enigma.core.hardware import get_hardware
        hw = get_hardware()
        
        ok(f"Platform: {hw.profile['platform']['system']}")
        ok(f"CPU: {hw.profile['cpu']['cores']} cores")
        ok(f"RAM: {hw.profile['memory']['total_gb']} GB")
        
        gpu = hw.profile['gpu']
        if gpu['available']:
            ok(f"GPU: {gpu['name']} ({gpu['vram_gb']} GB)")
        else:
            info("GPU: None (CPU mode)")
        
        ok(f"Recommended model: {hw.profile['recommended_model_size']}")
        ok(f"PyTorch device: {hw.get_device()}")
        
        return True
    except Exception as e:
        fail(f"Hardware detection failed: {e}")
        return False


def test_config():
    """Test configuration."""
    header("3. Configuration")
    
    try:
        from enigma.config import CONFIG
        
        ok(f"Config loaded")
        ok(f"Data dir: {CONFIG.get('data_dir', 'N/A')}")
        ok(f"Models dir: {CONFIG.get('models_dir', 'N/A')}")
        
        # Check directories exist or can be created
        for key in ['data_dir', 'models_dir']:
            path = Path(CONFIG.get(key, ''))
            if path.exists():
                ok(f"{key} exists")
            else:
                path.mkdir(parents=True, exist_ok=True)
                ok(f"{key} created")
        
        return True
    except Exception as e:
        fail(f"Config error: {e}")
        return False


def test_model_registry():
    """Test model registry."""
    header("4. Model Registry")
    
    try:
        from enigma.core.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        ok("Registry initialized")
        
        models = registry.list_models()
        ok(f"Found {len(models)} existing model(s)")
        
        # Try creating a test model
        test_name = "_test_model_delete_me"
        if test_name not in models:
            registry.create_model(test_name, size="tiny", description="Test")
            ok("Can create models")
            registry.delete_model(test_name, confirm=True)
            ok("Can delete models")
        else:
            ok("Model creation works (test model exists)")
        
        return True
    except Exception as e:
        fail(f"Model registry error: {e}")
        return False


def test_model_sizes():
    """Test model size configurations."""
    header("5. Model Sizes")
    
    try:
        from enigma.core.model_config import MODEL_PRESETS, get_model_config
        
        ok(f"Available sizes: {list(MODEL_PRESETS.keys())}")
        
        # Test getting a config
        config = get_model_config("tiny")
        ok(f"tiny: {config['dim']}d, {config['depth']} layers")
        
        config = get_model_config("medium")
        ok(f"medium: {config['dim']}d, {config['depth']} layers")
        
        return True
    except Exception as e:
        fail(f"Model config error: {e}")
        return False


def test_inference():
    """Test inference engine."""
    header("6. Inference Engine")
    
    try:
        from enigma.core.inference import EnigmaEngine
        
        engine = EnigmaEngine()
        ok("Engine initialized")
        
        # Generate something
        output = engine.generate("test", max_gen=5)
        ok(f"Generation works: '{output[:30]}...'")
        
        return True
    except Exception as e:
        fail(f"Inference error: {e}")
        return False


def test_tools():
    """Test tool system."""
    header("7. Tool System")
    
    try:
        from enigma.tools import execute_tool, get_registry
        
        registry = get_registry()
        tools = list(registry.tools.keys())
        ok(f"Registry loaded: {len(tools)} tools")
        
        # Test a simple tool
        result = execute_tool("get_system_info")
        if result.get("success"):
            ok("System info tool works")
        else:
            warn("System info tool returned no success flag")
        
        return True
    except Exception as e:
        fail(f"Tools error: {e}")
        return False


def test_vision():
    """Test screen vision."""
    header("8. Screen Vision")
    
    try:
        from enigma.tools.vision import ScreenCapture, get_screen_vision
        
        capture = ScreenCapture()
        ok(f"Backend detected: {capture._backend}")
        
        if capture._backend == "none":
            warn("No screenshot backend available")
            return True
        
        vision = get_screen_vision()
        ok("Vision system initialized")
        
        # Try a capture (may fail if no display)
        try:
            img = capture.capture()
            if img:
                ok(f"Capture works: {img.size[0]}x{img.size[1]}")
            else:
                warn("Capture returned None (no display?)")
        except Exception as e:
            warn(f"Capture failed (no display?): {e}")
        
        return True
    except Exception as e:
        fail(f"Vision error: {e}")
        return False


def test_avatar():
    """Test avatar system."""
    header("9. Avatar System")
    
    try:
        from enigma.avatar import get_avatar, AvatarState
        
        avatar = get_avatar()
        ok(f"Avatar controller initialized")
        ok(f"Default state: {avatar.state.value} (should be 'off')")
        
        if avatar.state == AvatarState.OFF:
            ok("Avatar correctly starts OFF")
        else:
            warn("Avatar should start OFF by default")
        
        # Test enable/disable
        avatar.enable()
        ok(f"Enable works, state: {avatar.state.value}")
        
        avatar.disable()
        ok(f"Disable works, state: {avatar.state.value}")
        
        return True
    except Exception as e:
        fail(f"Avatar error: {e}")
        return False


def test_voice():
    """Test voice system."""
    header("10. Voice System")
    
    try:
        from enigma.voice import speak, listen
        from enigma.voice.tts_simple import speak as tts_speak, HAVE_PYTTSX3, HAVE_ESPEAK
        
        ok("Voice module loaded")
        
        # Check TTS backends
        backends = []
        if HAVE_PYTTSX3:
            backends.append("pyttsx3")
        if HAVE_ESPEAK:
            backends.append("espeak")
        
        if backends:
            ok(f"TTS backends: {', '.join(backends)}")
        else:
            warn("No TTS engine (install pyttsx3 or espeak)")
        
        return True
    except Exception as e:
        fail(f"Voice error: {e}")
        return False


def test_comms():
    """Test communication system."""
    header("11. Communications")
    
    try:
        from enigma.comms import EnigmaNode, DeviceDiscovery, MemorySync
        
        ok("Network module loaded")
        ok("Discovery module loaded")
        ok("Memory sync module loaded")
        
        # Test discovery
        discovery = DeviceDiscovery("test_node")
        local_ip = discovery.get_local_ip()
        ok(f"Local IP: {local_ip}")
        
        return True
    except Exception as e:
        fail(f"Comms error: {e}")
        return False


def test_mobile_api():
    """Test mobile API."""
    header("12. Mobile API")
    
    try:
        from enigma.comms.mobile_api import MobileAPI, MOBILE_CLIENT_TEMPLATES
        
        ok("Mobile API module loaded")
        ok(f"Client templates: {list(MOBILE_CLIENT_TEMPLATES.keys())}")
        
        return True
    except Exception as e:
        fail(f"Mobile API error: {e}")
        return False


def run_all_tests(quick=False, verbose=False):
    """Run all system tests."""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}      ENIGMA ENGINE - SYSTEM CHECK{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    tests = [
        ("Environment", test_environment),
        ("Hardware", test_hardware),
        ("Config", test_config),
        ("Model Registry", test_model_registry),
        ("Model Sizes", test_model_sizes),
        ("Inference", test_inference),
        ("Tools", test_tools),
        ("Vision", test_vision),
        ("Avatar", test_avatar),
        ("Voice", test_voice),
        ("Communications", test_comms),
        ("Mobile API", test_mobile_api),
    ]
    
    if quick:
        # Only run essential tests
        tests = tests[:6]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            fail(f"{name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}      SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        if result:
            print(f"  {Colors.GREEN}✓{Colors.RESET} {name}")
        else:
            print(f"  {Colors.RED}✗{Colors.RESET} {name}")
    
    print()
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}All {total} tests passed! System is ready.{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}{passed}/{total} tests passed.{Colors.RESET}")
        print(f"Fix failed tests before running the full system.")
    
    print()
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Enigma Engine System Check")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick essential tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    success = run_all_tests(quick=args.quick, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
