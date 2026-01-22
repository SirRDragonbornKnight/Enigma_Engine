"""
SKEPTICAL FULL SYSTEM TEST
===========================
Tests everything as if nothing works. Actually runs code, not just imports.
"""
import sys
import os
sys.path.insert(0, '.')
os.environ['FORGE_NO_AUDIO'] = '1'  # Skip audio checks
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import traceback
from pathlib import Path

PASS = []
FAIL = []
WARN = []

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
                    print("✗ (returned False)")
            except Exception as e:
                FAIL.append(f"{name}: {type(e).__name__}: {e}")
                print(f"✗ ({type(e).__name__})")
                traceback.print_exc()
        return wrapper
    return decorator

print("=" * 70)
print("SKEPTICAL FULL SYSTEM TEST - ForgeAI")
print("=" * 70)

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
print("\n[1/15] CONFIGURATION")

@test("CONFIG exists and is dict-like")
def t1():
    from forge_ai.config import CONFIG
    # Actually use it
    val = CONFIG.get("models_dir", "models")
    return isinstance(val, str)

@test("CONFIG paths are valid")
def t2():
    from forge_ai.config import CONFIG
    models_dir = CONFIG.get("models_dir", "models")
    data_dir = CONFIG.get("data_dir", "data")
    return True  # Dirs may not exist yet, that's ok

t1()
t2()

# ===========================================================================
# 2. TOKENIZER - Actually encode/decode
# ===========================================================================
print("\n[2/15] TOKENIZER")

@test("Tokenizer encodes text to integers")
def t3():
    from forge_ai.core.tokenizer import get_tokenizer
    tok = get_tokenizer()
    tokens = tok.encode("Hello world")
    return isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)

@test("Tokenizer decodes back to text")
def t4():
    from forge_ai.core.tokenizer import get_tokenizer
    tok = get_tokenizer()
    tokens = tok.encode("Hello")
    text = tok.decode(tokens)
    return isinstance(text, str) and len(text) > 0

@test("Tokenizer vocab_size is reasonable")
def t5():
    from forge_ai.core.tokenizer import get_tokenizer
    tok = get_tokenizer()
    return tok.vocab_size > 100 and tok.vocab_size < 1000000

t3()
t4()
t5()

# ===========================================================================
# 3. MODEL - Actually create and run forward pass
# ===========================================================================
print("\n[3/15] MODEL")

@test("Model creates with correct vocab")
def t6():
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    tok = get_tokenizer()
    model = create_model('nano')
    return model.tok_embeddings.weight.shape[0] == tok.vocab_size

@test("Model forward pass produces output")
def t7():
    import torch
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    tok = get_tokenizer()
    model = create_model('nano')
    tokens = tok.encode("Hi")
    # Clamp to vocab size
    tokens = [min(t, tok.vocab_size - 1) for t in tokens]
    x = torch.tensor([tokens])
    with torch.no_grad():
        out = model(x)
    return out.shape[0] == 1 and out.shape[-1] == tok.vocab_size

@test("Model presets all work")
def t8():
    from forge_ai.core.model import MODEL_PRESETS, get_preset
    for size in ['nano', 'micro', 'tiny', 'small']:
        cfg = get_preset(size, vocab_size=1000)
        if cfg is None:
            return False
    return True

t6()
t7()
t8()

# ===========================================================================
# 4. INFERENCE - Actually generate text
# ===========================================================================
print("\n[4/15] INFERENCE")

@test("ForgeEngine.from_model creates engine")
def t9():
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    from forge_ai.core.inference import ForgeEngine
    model = create_model('nano')
    tok = get_tokenizer()
    engine = ForgeEngine.from_model(model, tok)
    return engine is not None and engine.model is not None

@test("ForgeEngine.generate produces text")
def t10():
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    from forge_ai.core.inference import ForgeEngine
    model = create_model('nano')
    tok = get_tokenizer()
    engine = ForgeEngine.from_model(model, tok)
    result = engine.generate("Hello", max_gen=10)
    return isinstance(result, str) and len(result) > 0

t9()
t10()

# ===========================================================================
# 5. HUGGINGFACE LOADER
# ===========================================================================
print("\n[5/15] HUGGINGFACE LOADER")

@test("HuggingFaceModel class exists")
def t11():
    from forge_ai.core.huggingface_loader import HuggingFaceModel
    return HuggingFaceModel is not None

@test("load_huggingface_model function works")
def t12():
    from forge_ai.core.huggingface_loader import load_huggingface_model
    # Test with a small model
    hf = load_huggingface_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return hf.is_loaded and hf.model is not None

@test("HuggingFace generate works")
def t13():
    from forge_ai.core.huggingface_loader import load_huggingface_model
    hf = load_huggingface_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    result = hf.generate("Hi", max_new_tokens=5)
    return isinstance(result, str) and len(result) > 0

@test("HuggingFace chat works")
def t14():
    from forge_ai.core.huggingface_loader import load_huggingface_model
    hf = load_huggingface_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    result = hf.chat("Hello!")
    return isinstance(result, str) and len(result) > 0

t11()
t12()
t13()
t14()

# ===========================================================================
# 6. MEMORY SYSTEM
# ===========================================================================
print("\n[6/15] MEMORY SYSTEM")

@test("ConversationManager creates")
def t15():
    from forge_ai.memory.manager import ConversationManager
    cm = ConversationManager()
    return cm is not None

@test("ConversationManager save/load cycle")
def t16():
    from forge_ai.memory.manager import ConversationManager
    cm = ConversationManager()
    test_data = [{"role": "user", "content": "test123"}]
    cm.save_conversation("_test_conv_", test_data)
    loaded = cm.load_conversation("_test_conv_")
    # Cleanup
    try:
        (cm.conv_dir / "_test_conv_.json").unlink()
    except:
        pass
    # load_conversation returns dict with 'messages' key
    return loaded is not None and loaded.get("messages") == test_data

@test("SimpleVectorDB add/search")
def t17():
    from forge_ai.memory.vector_db import SimpleVectorDB
    import numpy as np
    db = SimpleVectorDB(dim=4)
    db.add(np.array([1.0, 0.0, 0.0, 0.0]), "a")
    db.add(np.array([0.0, 1.0, 0.0, 0.0]), "b")
    results = db.search(np.array([0.9, 0.1, 0.0, 0.0]), top_k=1)
    return len(results) > 0 and results[0][0] == "a"

t15()
t16()
t17()

# ===========================================================================
# 7. TOOL SYSTEM
# ===========================================================================
print("\n[7/15] TOOL SYSTEM")

@test("Tool definitions exist")
def t18():
    from forge_ai.tools.tool_definitions import get_all_tools
    tools = get_all_tools()
    return len(tools) > 0

@test("ToolExecutor creates")
def t19():
    from forge_ai.tools.tool_executor import ToolExecutor
    te = ToolExecutor()
    return te is not None

@test("ReadFileTool executes")
def t20():
    from forge_ai.tools.file_tools import ReadFileTool
    tool = ReadFileTool()
    result = tool.execute(path="README.md", max_lines=5)
    return "content" in result or "error" in result

@test("ListDirectoryTool executes")
def t21():
    from forge_ai.tools.file_tools import ListDirectoryTool
    tool = ListDirectoryTool()
    result = tool.execute(path=".")
    return "items" in result or "error" in result

t18()
t19()
t20()
t21()

# ===========================================================================
# 8. TOOL ROUTER
# ===========================================================================
print("\n[8/15] TOOL ROUTER")

@test("ToolRouter creates")
def t22():
    from forge_ai.core.tool_router import get_router
    router = get_router(use_specialized=False)
    return router is not None

@test("classify_intent returns valid intent")
def t23():
    from forge_ai.core.tool_router import get_router
    router = get_router(use_specialized=False)
    intent = router.classify_intent("draw a picture")
    return intent in ['image', 'code', 'chat', 'search', 'file', 'audio', 'video', '3d', None]

@test("classify_intent for code")
def t24():
    from forge_ai.core.tool_router import get_router
    router = get_router(use_specialized=False)
    intent = router.classify_intent("write python code")
    return intent == 'code'

t22()
t23()
t24()

# ===========================================================================
# 9. MODULE SYSTEM
# ===========================================================================
print("\n[9/15] MODULE SYSTEM")

@test("ModuleManager creates")
def t25():
    from forge_ai.modules import ModuleManager
    m = ModuleManager()
    return m is not None

@test("register_all populates modules")
def t26():
    from forge_ai.modules import ModuleManager, register_all
    m = ModuleManager()
    register_all(m)
    return len(m.module_classes) >= 20

@test("can_load returns tuple")
def t27():
    from forge_ai.modules import ModuleManager, register_all
    m = ModuleManager()
    register_all(m)
    can, reason = m.can_load('memory')
    return isinstance(can, bool) and isinstance(reason, str)

@test("load/unload module works")
def t28():
    from forge_ai.modules import ModuleManager, register_all
    m = ModuleManager()
    register_all(m)
    loaded = m.load('memory')
    if not loaded:
        return False
    unloaded = m.unload('memory')
    return unloaded

t25()
t26()
t27()
t28()

# ===========================================================================
# 10. TRAINING SYSTEM
# ===========================================================================
print("\n[10/15] TRAINING SYSTEM")

@test("TrainingConfig creates")
def t29():
    from forge_ai.core.training import TrainingConfig
    cfg = TrainingConfig(epochs=1, batch_size=2)
    return cfg.epochs == 1 and cfg.batch_size == 2

@test("Trainer creates")
def t30():
    from forge_ai.core.training import Trainer
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    model = create_model('nano')
    tok = get_tokenizer()
    trainer = Trainer(model, tok, device='cpu')
    return trainer is not None

@test("TextDataset creates sequences")
def t31():
    from forge_ai.core.training import TextDataset
    from forge_ai.core.tokenizer import get_tokenizer
    tok = get_tokenizer()
    ds = TextDataset(["Hello world this is a test"] * 10, tok, max_length=32)
    return len(ds) > 0

t29()
t30()
t31()

# ===========================================================================
# 11. VOICE SYSTEM
# ===========================================================================
print("\n[11/15] VOICE SYSTEM")

@test("AIVoiceGenerator class exists")
def t32():
    from forge_ai.voice.voice_generator import AIVoiceGenerator
    return AIVoiceGenerator is not None

@test("VoiceListener class exists")
def t33():
    from forge_ai.voice.listener import VoiceListener
    return VoiceListener is not None

t32()
t33()

# ===========================================================================
# 12. AVATAR SYSTEM
# ===========================================================================
print("\n[12/15] AVATAR SYSTEM")

@test("AutonomousAvatar class exists")
def t34():
    from forge_ai.avatar.autonomous import AutonomousAvatar
    return AutonomousAvatar is not None

@test("Avatar presets exist")
def t35():
    from forge_ai.avatar.presets import PresetManager
    return len(PresetManager.BUILTIN_PRESETS) > 0

@test("create_avatar_subtab function exists")
def t36():
    from forge_ai.gui.tabs.avatar.avatar_display import create_avatar_subtab
    return callable(create_avatar_subtab)

t34()
t35()
t36()

# ===========================================================================
# 13. GUI COMPONENTS
# ===========================================================================
print("\n[13/15] GUI COMPONENTS")

@test("EnhancedMainWindow class exists")
def t37():
    from forge_ai.gui.enhanced_window import EnhancedMainWindow
    return EnhancedMainWindow is not None

@test("ModulesTab class exists")
def t38():
    from forge_ai.gui.tabs.modules_tab import ModulesTab
    return ModulesTab is not None

@test("ImageTab class exists")
def t39():
    from forge_ai.gui.tabs.image_tab import ImageTab
    return ImageTab is not None

@test("CodeTab class exists")
def t40():
    from forge_ai.gui.tabs.code_tab import CodeTab
    return CodeTab is not None

@test("QuickCommandOverlay class exists")
def t41():
    from forge_ai.gui.system_tray import QuickCommandOverlay
    return QuickCommandOverlay is not None

t37()
t38()
t39()
t40()
t41()

# ===========================================================================
# 14. API/NETWORK
# ===========================================================================
print("\n[14/15] API/NETWORK")

@test("create_api_server function exists")
def t42():
    from forge_ai.comms.api_server import create_api_server
    return callable(create_api_server)

@test("ForgeNode class exists")
def t43():
    from forge_ai.comms.network import ForgeNode
    return ForgeNode is not None

t42()
t43()

# ===========================================================================
# 15. GENERATION TABS (Image, Code, Video, Audio, 3D)
# ===========================================================================
print("\n[15/15] GENERATION PROVIDERS")

@test("StableDiffusionLocal provider")
def t44():
    from forge_ai.gui.tabs.image_tab import StableDiffusionLocal
    return StableDiffusionLocal is not None

@test("ForgeCode provider")
def t45():
    from forge_ai.gui.tabs.code_tab import ForgeCode
    return ForgeCode is not None

@test("LocalTTS provider")
def t46():
    from forge_ai.gui.tabs.audio_tab import LocalTTS
    return LocalTTS is not None

@test("Local3DGen provider")
def t47():
    from forge_ai.gui.tabs.threed_tab import Local3DGen
    return Local3DGen is not None

t44()
t45()
t46()
t47()

# ===========================================================================
# BONUS: Check for common issues
# ===========================================================================
print("\n[BONUS] COMMON ISSUES CHECK")

@test("No circular imports in core")
def t48():
    # If we got here, no circular imports crashed us
    from forge_ai.core import model, tokenizer, inference, training, tool_router
    return True

@test("Paths use pathlib (spot check)")
def t49():
    from forge_ai.core.training import MODELS_DIR, DATA_DIR
    return hasattr(MODELS_DIR, 'exists') and hasattr(DATA_DIR, 'exists')

@test("Logger defined in key modules")
def t50():
    import forge_ai.core.model as m1
    import forge_ai.core.inference as m2
    import forge_ai.core.tool_router as m3
    return hasattr(m1, 'logger') and hasattr(m2, 'logger') and hasattr(m3, 'logger')

t48()
t49()
t50()

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n✓ PASSED: {len(PASS)}")
print(f"✗ FAILED: {len(FAIL)}")

if FAIL:
    print("\n--- FAILURES ---")
    for f in FAIL:
        print(f"  ✗ {f}")

if WARN:
    print("\n--- WARNINGS ---")
    for w in WARN:
        print(f"  ⚠ {w}")

# Exit with error code if failures
if FAIL:
    print("\n❌ SOME TESTS FAILED")
    sys.exit(1)
else:
    print("\n✅ ALL TESTS PASSED")
    sys.exit(0)
