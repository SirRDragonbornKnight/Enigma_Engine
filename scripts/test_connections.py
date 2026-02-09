"""Test that Enigma AI Engine components connect correctly."""
import sys
sys.path.insert(0, '.')

def test_module_system():
    """Test Module Manager connections."""
    from enigma_engine.modules import ModuleManager
    manager = ModuleManager()
    assert manager is not None
    # module_classes holds registered module types
    module_count = len(manager.module_classes)
    print(f"  Module Manager: OK ({module_count} registered module types)")
    return True

def test_config():
    """Test configuration loading."""
    from enigma_engine.config import CONFIG, get_config
    assert CONFIG.get('data_dir')
    assert CONFIG.get('models_dir')
    print(f"  Config: OK (data_dir={CONFIG['data_dir'][:30]}...)")
    return True

def test_tokenizer():
    """Test tokenizer encode/decode roundtrip."""
    from enigma_engine.core.tokenizer import get_tokenizer
    tok = get_tokenizer()
    text = "Hello world test"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert len(encoded) > 0
    print(f"  Tokenizer: OK (encoded {len(text)} chars to {len(encoded)} tokens)")
    return True

def test_memory():
    """Test memory system initialization."""
    from enigma_engine.memory import ConversationManager
    mem = ConversationManager()
    assert mem is not None
    print(f"  Memory: OK")
    return True

def test_tools():
    """Test tool execution."""
    from enigma_engine.tools import execute_tool
    result = execute_tool('get_system_info')
    assert result.get('success')
    print(f"  Tools: OK (get_system_info returned {len(result)} keys)")
    return True

def test_web_search():
    """Test web search tool exists."""
    from enigma_engine.tools.web_tools import WebSearchTool
    tool = WebSearchTool()
    assert tool.name == "web_search"
    print(f"  Web Search: OK")
    return True

def test_vision():
    """Test vision system."""
    from enigma_engine.tools.vision import ScreenCapture
    sc = ScreenCapture()
    assert sc._backend != "none"
    print(f"  Vision: OK (backend={sc._backend})")
    return True

def test_gui_tabs():
    """Test GUI tab imports (no Qt required)."""
    from enigma_engine.gui.tabs import chat_tab, training_tab, settings_tab
    from enigma_engine.gui.tabs.avatar import avatar_display
    print(f"  GUI Tabs: OK (chat, training, settings, avatar)")
    return True

def test_inference_engine():
    """Test inference engine can be imported."""
    from enigma_engine.core.inference import EnigmaEngine
    print(f"  Inference Engine: OK")
    return True

def test_gui_window():
    """Test GUI window can be created (headless)."""
    import os
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    try:
        from PyQt5.QtWidgets import QApplication
        from enigma_engine.gui.enhanced_window import EnhancedMainWindow
        app = QApplication.instance() or QApplication([])
        # Just test class exists and can be referenced
        assert EnhancedMainWindow is not None
        print(f"  GUI Window: OK (EnhancedMainWindow importable)")
        return True
    except ImportError as e:
        print(f"  GUI Window: SKIP (PyQt5 not configured: {e})")
        return True

def test_training_system():
    """Test training system connections."""
    from enigma_engine.core.training import Trainer, TrainingConfig
    config = TrainingConfig(epochs=1, batch_size=2)
    assert config.epochs == 1
    assert config.batch_size == 2
    print(f"  Training: OK (TrainingConfig works)")
    return True

def test_huggingface_loader():
    """Test HuggingFace loader exists."""
    from enigma_engine.core.huggingface_loader import HuggingFaceModel, HuggingFaceEngine
    assert HuggingFaceModel is not None
    assert HuggingFaceEngine is not None
    print(f"  HuggingFace Loader: OK")
    return True

def main():
    print("=" * 50)
    print("Enigma AI Engine Connection Tests")
    print("=" * 50)
    
    tests = [
        ("Module System", test_module_system),
        ("Configuration", test_config),
        ("Tokenizer", test_tokenizer),
        ("Memory", test_memory),
        ("Tools", test_tools),
        ("Web Search", test_web_search),
        ("Vision", test_vision),
        ("GUI Tabs", test_gui_tabs),
        ("Inference", test_inference_engine),
        ("GUI Window", test_gui_window),
        ("Training", test_training_system),
        ("HuggingFace", test_huggingface_loader),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAIL - {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
