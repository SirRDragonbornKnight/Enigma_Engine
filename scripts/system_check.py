#!/usr/bin/env python3
"""
Pre-Training System Check
=========================

Run this BEFORE training to make sure everything works!
This tests each component independently so you know what to expect.

Run: python scripts/system_check.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_test(name, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        print(f"         {details}")
    return passed

def main():
    print_header("ForgeAI Pre-Training System Check")
    print("Testing each component to see what works...")
    
    results = {}
    
    # =========================================================================
    # TEST 1: Core Imports
    # =========================================================================
    print_header("1. Core Module Imports")
    
    try:
        import torch
        results['pytorch'] = print_test("PyTorch", True, f"v{torch.__version__}")
    except ImportError as e:
        results['pytorch'] = print_test("PyTorch", False, str(e))
    
    try:
        from forge_ai.core.model import create_model, MODEL_PRESETS
        results['model'] = print_test("Forge Model", True, f"{len(MODEL_PRESETS)} presets available")
    except Exception as e:
        results['model'] = print_test("Forge Model", False, str(e))
    
    try:
        from forge_ai.core.tokenizer import get_tokenizer
        tok = get_tokenizer()
        test_encode = tok.encode("Hello world")
        results['tokenizer'] = print_test("Tokenizer", True, f"Encoded 'Hello world' -> {len(test_encode)} tokens")
    except Exception as e:
        results['tokenizer'] = print_test("Tokenizer", False, str(e))
    
    # =========================================================================
    # TEST 2: Model Creation (without training)
    # =========================================================================
    print_header("2. Model Creation Test")
    
    try:
        from forge_ai.core.model import create_model
        import torch
        
        # Create a tiny model to test
        print("  Creating 'nano' model (smallest)...")
        model = create_model('nano')
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        results['model_create'] = print_test("Model Creation", True, f"nano model has {params:,} parameters")
        
        # Test forward pass
        print("  Testing forward pass...")
        dummy_input = torch.randint(0, 1000, (1, 10))  # batch=1, seq=10
        with torch.no_grad():
            output = model(dummy_input)
        results['forward_pass'] = print_test("Forward Pass", True, f"Output shape: {output.shape}")
        
        del model  # Free memory
        
    except Exception as e:
        results['model_create'] = print_test("Model Creation", False, str(e))
        results['forward_pass'] = False
    
    # =========================================================================
    # TEST 3: Tool Router (keyword-based routing)
    # =========================================================================
    print_header("3. Tool Router Test")
    
    try:
        from forge_ai.core.tool_router import get_router, classify_intent
        
        # Test intent classification
        test_cases = [
            ("Draw me a cat", "image"),
            ("Explain quantum physics", "chat"),
            ("Write a Python function", "code"),
            ("Search the web for news", "web_search"),
        ]
        
        router = get_router()
        all_passed = True
        
        for prompt, expected in test_cases:
            result = classify_intent(prompt)
            matched = expected in str(result).lower()
            if not matched:
                print(f"    '{prompt}' -> Got {result}, expected {expected}")
                all_passed = False
        
        results['router'] = print_test("Tool Router", all_passed, 
            "Routes: image, code, chat, web correctly" if all_passed else "Some routes failed")
        
    except Exception as e:
        results['router'] = print_test("Tool Router", False, str(e))
    
    # =========================================================================
    # TEST 4: Tool Executor
    # =========================================================================
    print_header("4. Tool Executor Test")
    
    try:
        from forge_ai.tools.tool_executor import ToolExecutor
        
        executor = ToolExecutor()
        results['executor_init'] = print_test("Tool Executor Init", True)
        
        # Test a simple tool (list_directory)
        test_result = executor.execute("list_directory", {"path": "."})
        success = test_result.get("success", False) or "items" in test_result
        results['executor_run'] = print_test("Tool Execution (list_dir)", success,
            f"Found {len(test_result.get('items', []))} items" if success else str(test_result))
        
    except Exception as e:
        results['executor_init'] = print_test("Tool Executor", False, str(e))
    
    # =========================================================================
    # TEST 5: Image Generation Check
    # =========================================================================
    print_header("5. Image Generation Check")
    
    # Check if Stable Diffusion or alternatives are available
    try:
        from diffusers import StableDiffusionPipeline
        results['sd_available'] = print_test("Stable Diffusion (diffusers)", True, 
            "Installed - but needs model download (~4GB)")
    except ImportError:
        results['sd_available'] = print_test("Stable Diffusion (diffusers)", False, 
            "Not installed. Run: pip install diffusers transformers accelerate")
    
    # Check for API-based alternatives
    import os
    has_openai = bool(os.environ.get('OPENAI_API_KEY'))
    has_replicate = bool(os.environ.get('REPLICATE_API_TOKEN'))
    
    if has_openai:
        results['image_api'] = print_test("OpenAI DALL-E API", True, "API key found")
    elif has_replicate:
        results['image_api'] = print_test("Replicate API", True, "API token found")
    else:
        results['image_api'] = print_test("Image API Keys", False, 
            "No OPENAI_API_KEY or REPLICATE_API_TOKEN set. Local SD or API needed for images.")
    
    # =========================================================================
    # TEST 6: Voice/TTS Check
    # =========================================================================
    print_header("6. Voice/TTS Check")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        results['tts_local'] = print_test("Local TTS (pyttsx3)", True, "Ready to speak")
        engine.stop()
    except Exception as e:
        results['tts_local'] = print_test("Local TTS (pyttsx3)", False, str(e))
    
    # =========================================================================
    # TEST 7: Memory/Conversation System
    # =========================================================================
    print_header("7. Memory System Check")
    
    try:
        from forge_ai.memory.manager import ConversationManager
        
        manager = ConversationManager()
        manager.add_message("user", "Hello test")
        manager.add_message("assistant", "Hi there!")
        history = manager.get_history()
        
        results['memory'] = print_test("Conversation Memory", len(history) == 2,
            f"Stored {len(history)} messages")
        
    except Exception as e:
        results['memory'] = print_test("Conversation Memory", False, str(e))
    
    # =========================================================================
    # TEST 8: Web Tools Check
    # =========================================================================
    print_header("8. Web Tools Check")
    
    try:
        import requests
        response = requests.get("https://httpbin.org/get", timeout=5)
        results['web_tools'] = print_test("HTTP Requests", response.status_code == 200,
            "Can fetch web pages")
    except Exception as e:
        results['web_tools'] = print_test("HTTP Requests", False, str(e))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n  Tests passed: {passed}/{total}")
    print()
    
    # Critical vs optional
    critical = ['pytorch', 'model', 'tokenizer', 'model_create', 'forward_pass']
    critical_passed = sum(1 for k in critical if results.get(k))
    
    if critical_passed == len(critical):
        print("  [OK] CORE TRAINING READY")
        print("       Your system can train AI models!")
    else:
        print("  [!!] CORE ISSUES")
        print("       Fix these before training:")
        for k in critical:
            if not results.get(k):
                print(f"       - {k}")
    
    print()
    
    # Tool availability
    tool_tests = ['router', 'executor_init', 'executor_run', 'web_tools']
    tools_passed = sum(1 for k in tool_tests if results.get(k))
    
    if tools_passed == len(tool_tests):
        print("  [OK] BASIC TOOLS READY")
        print("       File operations, web search, routing all work!")
    else:
        print("  [!] SOME TOOLS UNAVAILABLE")
    
    # Image generation
    if results.get('sd_available') or results.get('image_api'):
        print("  [OK] IMAGE GENERATION AVAILABLE")
    else:
        print("  [!] IMAGE GENERATION NOT SET UP")
        print("       Either install diffusers or set API keys")
    
    print()
    print("=" * 60)
    
    # Final recommendation
    if critical_passed == len(critical):
        print("""
  RECOMMENDATION: You're ready to train!
  
  Start with a quick test:
    python run.py --train --epochs 5 --model-size pi_4
  
  What to expect after training:
    - The AI WILL respond to prompts
    - Responses will be LIMITED (small model)
    - Tools WILL work (they're separate systems)
    - Quality improves with more data + epochs
""")
    else:
        print("""
  RECOMMENDATION: Fix the core issues first.
  
  Missing dependencies? Try:
    pip install torch psutil tiktoken
""")

if __name__ == "__main__":
    main()
