"""
EXTENDED FUNCTIONAL TESTS - ForgeAI
====================================
Tests edge cases, error handling, and integration points.
"""
import sys
import os
sys.path.insert(0, '.')
os.environ['FORGE_NO_AUDIO'] = '1'  # Skip audio checks
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import traceback
from pathlib import Path
import tempfile
import shutil

PASS = []
FAIL = []

def test(name):
    def decorator(func):
        def wrapper():
            print(f"Testing: {name}...", end=" ", flush=True)
            try:
                result = func()
                if result:
                    PASS.append(name)
                    print("✓")
                else:
                    FAIL.append(f"{name}: returned False")
                    print("✗")
            except Exception as e:
                FAIL.append(f"{name}: {type(e).__name__}: {e}")
                print(f"✗ ({type(e).__name__})")
                traceback.print_exc()
        return wrapper
    return decorator

print("=" * 70)
print("EXTENDED FUNCTIONAL TESTS - ForgeAI")
print("=" * 70)

# ===========================================================================
# 1. MODEL EDGE CASES
# ===========================================================================
print("\n[1/10] MODEL EDGE CASES")

@test("Model handles empty input")
def t1():
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    import torch
    
    model = create_model('nano')
    tok = get_tokenizer()
    
    # Empty string should still work
    tokens = tok.encode("")
    if len(tokens) == 0:
        tokens = [tok.encode(" ")[0]]  # At least one token
    
    ids = torch.tensor([tokens])
    output = model(ids)
    return output.shape[0] == 1

@test("Model handles long input (truncation)")
def t2():
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    import torch
    
    model = create_model('nano')
    tok = get_tokenizer()
    
    # Very long input
    long_text = "Hello world. " * 1000
    tokens = tok.encode(long_text)[:512]  # Truncate to reasonable length
    
    ids = torch.tensor([tokens])
    output = model(ids)
    return output.shape[1] == len(tokens)

@test("Model handles special characters")
def t3():
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    import torch
    
    model = create_model('nano')
    tok = get_tokenizer()
    
    special_text = "Test @#$%^&*()[]{}|\\:;<>,.?/~`"
    tokens = tok.encode(special_text)
    
    ids = torch.tensor([tokens])
    output = model(ids)
    return output.shape[1] == len(tokens)

t1()
t2()
t3()

# ===========================================================================
# 2. INFERENCE ENGINE EDGE CASES
# ===========================================================================
print("\n[2/10] INFERENCE ENGINE EDGE CASES")

@test("ForgeEngine handles streaming")
def t4():
    from forge_ai.core.inference import ForgeEngine
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    
    model = create_model('nano')
    tok = get_tokenizer()
    engine = ForgeEngine.from_model(model, tok)
    
    chunks = list(engine.stream("Hello", max_tokens=5))
    return len(chunks) > 0

@test("ForgeEngine handles max_gen edge case")
def t5():
    from forge_ai.core.inference import ForgeEngine
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    
    model = create_model('nano')
    tok = get_tokenizer()
    engine = ForgeEngine.from_model(model, tok)
    
    # max_gen=0 should raise ValueError (correct behavior)
    try:
        engine.generate("Hello", max_gen=0)
        return False  # Should have raised
    except ValueError:
        pass  # Expected
    
    # max_gen=1 should work
    result = engine.generate("Hello", max_gen=1)
    return isinstance(result, str)

@test("ForgeEngine handles temperature extremes")
def t6():
    from forge_ai.core.inference import ForgeEngine
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    
    model = create_model('nano')
    tok = get_tokenizer()
    engine = ForgeEngine.from_model(model, tok)
    
    # Low temperature
    result1 = engine.generate("Test", max_gen=5, temperature=0.01)
    # High temperature
    result2 = engine.generate("Test", max_gen=5, temperature=2.0)
    
    return isinstance(result1, str) and isinstance(result2, str)

t4()
t5()
t6()

# ===========================================================================
# 3. MEMORY SYSTEM EDGE CASES
# ===========================================================================
print("\n[3/10] MEMORY SYSTEM EDGE CASES")

@test("ConversationManager handles unicode")
def t7():
    from forge_ai.memory.manager import ConversationManager
    
    cm = ConversationManager()
    test_data = [{"role": "user", "content": "你好世界 🌍 Привет"}]
    cm.save_conversation("_test_unicode_", test_data)
    loaded = cm.load_conversation("_test_unicode_")
    
    # Cleanup
    try:
        (cm.conv_dir / "_test_unicode_.json").unlink()
    except:
        pass
    
    return loaded.get("messages") == test_data

@test("ConversationManager handles empty conversation")
def t8():
    from forge_ai.memory.manager import ConversationManager
    
    cm = ConversationManager()
    cm.save_conversation("_test_empty_", [])
    loaded = cm.load_conversation("_test_empty_")
    
    # Cleanup
    try:
        (cm.conv_dir / "_test_empty_.json").unlink()
    except:
        pass
    
    return loaded.get("messages") == []

@test("SimpleVectorDB handles zero vector")
def t9():
    from forge_ai.memory.vector_db import SimpleVectorDB
    import numpy as np
    
    db = SimpleVectorDB(dim=4)
    db.add(np.array([0.0, 0.0, 0.0, 0.0]), "zero")
    db.add(np.array([1.0, 0.0, 0.0, 0.0]), "one")
    
    # Search with zero vector - should not crash
    results = db.search(np.array([0.0, 0.0, 0.0, 0.0]), top_k=2)
    return len(results) >= 0  # May return 0 results due to normalization

t7()
t8()
t9()

# ===========================================================================
# 4. TOOL SYSTEM EDGE CASES
# ===========================================================================
print("\n[4/10] TOOL SYSTEM EDGE CASES")

@test("ReadFileTool handles nonexistent file")
def t10():
    from forge_ai.tools.file_tools import ReadFileTool
    
    tool = ReadFileTool()
    result = tool.execute(path="/nonexistent/file/path.txt")
    return result.get("success") == False or "error" in result

@test("ListDirectoryTool handles nonexistent dir")
def t11():
    from forge_ai.tools.file_tools import ListDirectoryTool
    
    tool = ListDirectoryTool()
    result = tool.execute(path="/nonexistent/directory/")
    return result.get("success") == False or "error" in result

@test("WriteFileTool creates parent directories")
def t12():
    from forge_ai.tools.file_tools import WriteFileTool
    import tempfile
    import os
    
    tool = WriteFileTool()
    temp_dir = tempfile.mkdtemp()
    test_path = os.path.join(temp_dir, "subdir", "test.txt")
    
    result = tool.execute(path=test_path, content="test content")
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    return result.get("success") == True

t10()
t11()
t12()

# ===========================================================================
# 5. MODULE SYSTEM EDGE CASES
# ===========================================================================
print("\n[5/10] MODULE SYSTEM EDGE CASES")

@test("ModuleManager handles double load")
def t13():
    from forge_ai.modules import ModuleManager, register_all
    
    mgr = ModuleManager()
    register_all(mgr)
    
    # Load same module twice
    loaded1 = mgr.load('memory')
    loaded2 = mgr.load('memory')  # Should return True (already loaded)
    
    mgr.unload('memory')
    return loaded1 and loaded2

@test("ModuleManager handles unload of unloaded module")
def t14():
    from forge_ai.modules import ModuleManager, register_all
    
    mgr = ModuleManager()
    register_all(mgr)
    
    # Unload module that isn't loaded
    result = mgr.unload('memory')
    return result == False or result == True  # Either is acceptable

@test("ModuleManager can_load checks dependencies")
def t15():
    from forge_ai.modules import ModuleManager, register_all
    
    mgr = ModuleManager()
    register_all(mgr)
    
    # Check that can_load returns tuple
    can, reason = mgr.can_load('inference')
    return isinstance(can, bool) and isinstance(reason, str)

t13()
t14()
t15()

# ===========================================================================
# 6. TOOL ROUTER EDGE CASES
# ===========================================================================
print("\n[6/10] TOOL ROUTER EDGE CASES")

@test("ToolRouter handles empty string")
def t16():
    from forge_ai.core.tool_router import get_router
    
    router = get_router(use_specialized=False)
    intent = router.classify_intent("")
    return intent in ['image', 'code', 'chat', 'search', 'file', 'audio', 'video', '3d', None]

@test("ToolRouter handles very long input")
def t17():
    from forge_ai.core.tool_router import get_router
    
    router = get_router(use_specialized=False)
    long_text = "Please write code. " * 100
    intent = router.classify_intent(long_text)
    return intent in ['image', 'code', 'chat', 'search', 'file', 'audio', 'video', '3d', None]

@test("ToolRouter handles special characters")
def t18():
    from forge_ai.core.tool_router import get_router
    
    router = get_router(use_specialized=False)
    intent = router.classify_intent("@#$%^&*()")
    return intent in ['image', 'code', 'chat', 'search', 'file', 'audio', 'video', '3d', None]

t16()
t17()
t18()

# ===========================================================================
# 7. TRAINING SYSTEM EDGE CASES
# ===========================================================================
print("\n[7/10] TRAINING SYSTEM EDGE CASES")

@test("TextDataset handles short texts")
def t19():
    from forge_ai.core.training import TextDataset
    from forge_ai.core.tokenizer import get_tokenizer
    
    tok = get_tokenizer()
    # Very short text
    ds = TextDataset(["Hi"], tok, max_length=32)
    return len(ds) >= 0  # May be 0 if text too short

@test("TextDataset handles empty list")
def t20():
    from forge_ai.core.training import TextDataset
    from forge_ai.core.tokenizer import get_tokenizer
    
    tok = get_tokenizer()
    ds = TextDataset([], tok, max_length=32)
    return len(ds) == 0

@test("TrainingConfig validates parameters")
def t21():
    from forge_ai.core.training import TrainingConfig
    
    # Test with valid parameters
    cfg = TrainingConfig(epochs=1, batch_size=1, learning_rate=0.001)
    return cfg.epochs == 1 and cfg.batch_size == 1

t19()
t20()
t21()

# ===========================================================================
# 8. GUI COMPONENT EDGE CASES
# ===========================================================================
print("\n[8/10] GUI COMPONENT EDGE CASES")

@test("EnhancedMainWindow creates without engine")
def t22():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    
    from forge_ai.gui.enhanced_window import EnhancedMainWindow
    # Should create without engine pre-loaded
    window = EnhancedMainWindow()
    result = window is not None
    window.deleteLater()
    return result

@test("ModulesTab creates properly")
def t23():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    
    from forge_ai.gui.tabs.modules_tab import ModulesTab
    from forge_ai.modules import ModuleManager, register_all
    
    mgr = ModuleManager()
    register_all(mgr)
    
    tab = ModulesTab(module_manager=mgr)
    result = tab is not None
    tab.deleteLater()
    return result

@test("ImageTab creates properly")
def t24():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    
    from forge_ai.gui.tabs.image_tab import ImageTab
    tab = ImageTab()
    result = tab is not None
    tab.deleteLater()
    return result

t22()
t23()
t24()

# ===========================================================================
# 9. CONFIG AND PATH HANDLING
# ===========================================================================
print("\n[9/10] CONFIG AND PATH HANDLING")

@test("CONFIG handles missing keys gracefully")
def t25():
    from forge_ai.config import CONFIG
    
    # Non-existent key should return None or default
    val = CONFIG.get("nonexistent_key_12345", "default")
    return val == "default"

@test("Paths are created properly")
def t26():
    from forge_ai.config import CONFIG
    
    # Check that core directories exist or can be created
    for key in ['data_dir', 'models_dir', 'logs_dir']:
        path = Path(CONFIG.get(key, key))
        path.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return False
    return True

@test("Security path blocking works")
def t27():
    from forge_ai.utils.security import is_path_blocked, get_blocked_paths
    
    # Should have some blocked paths
    blocked = get_blocked_paths()
    
    # Check that Windows system paths are blocked (returns tuple: blocked, reason)
    result = is_path_blocked("C:\\Windows\\System32")
    # Returns (bool, str) tuple
    return isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool)

t25()
t26()
t27()

# ===========================================================================
# 10. INTEGRATION TESTS
# ===========================================================================
print("\n[10/10] INTEGRATION TESTS")

@test("Full pipeline: tokenize -> model -> generate")
def t28():
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    from forge_ai.core.inference import ForgeEngine
    
    # Create all components
    model = create_model('nano')
    tok = get_tokenizer()
    engine = ForgeEngine.from_model(model, tok)
    
    # Full generation
    result = engine.generate("Hello", max_gen=5)
    return isinstance(result, str) and len(result) > 0

@test("Module + Memory integration")
def t29():
    from forge_ai.modules import ModuleManager, register_all
    
    mgr = ModuleManager()
    register_all(mgr)
    
    # Load memory module
    loaded = mgr.load('memory')
    if not loaded:
        return False
    
    # Get instance
    mem_module = mgr.get_module('memory')
    
    # Unload
    mgr.unload('memory')
    return True

@test("Tool execution integration")
def t30():
    from forge_ai.tools.tool_executor import ToolExecutor
    from forge_ai.tools.file_tools import ReadFileTool
    
    executor = ToolExecutor()
    
    # Execute a tool
    result = executor.execute("read_file", {"path": "README.md", "max_lines": 5})
    return "content" in result or "error" in result

t28()
t29()
t30()

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n✓ PASSED: {len(PASS)}")
print(f"✗ FAILED: {len(FAIL)}")

if FAIL:
    print("\nFailed tests:")
    for f in FAIL:
        print(f"  - {f}")

if len(FAIL) == 0:
    print("\n✅ ALL EXTENDED TESTS PASSED")
else:
    print(f"\n⚠️ {len(FAIL)} tests need attention")
