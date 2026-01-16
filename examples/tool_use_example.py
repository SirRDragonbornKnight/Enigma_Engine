#!/usr/bin/env python3
"""
Example: Using the Tool Use System

This example demonstrates how the ForgeAI AI can use tools naturally in conversation.

Steps:
1. Initialize inference engine with tool support
2. Load required modules
3. Have a conversation where AI uses tools
"""

from forge_ai.core.inference import ForgeEngine
from forge_ai.modules import ModuleManager
from forge_ai.tools import get_available_tools_for_prompt


def main():
    print("=" * 80)
    print("AI TESTER - TOOL USE EXAMPLE")
    print("=" * 80)
    print()
    
    # Initialize module manager
    print("1. Initializing module manager...")
    manager = ModuleManager()
    
    # Load core modules
    print("2. Loading core modules...")
    try:
        manager.load('model')
        manager.load('tokenizer')
        print("   [OK] Core modules loaded")
    except Exception as e:
        print(f"   [FAIL] Could not load modules: {e}")
        print("   Note: Modules may not be available in this environment")
    
    # Initialize inference engine with tools enabled
    print("3. Initializing inference engine with tool support...")
    try:
        engine = ForgeEngine(
            model_size="small",
            enable_tools=True,
            module_manager=manager
        )
        print("   [OK] Engine initialized with tool support")
    except Exception as e:
        print(f"   [FAIL] Could not initialize engine: {e}")
        return
    
    print()
    print("=" * 80)
    print("AVAILABLE TOOLS")
    print("=" * 80)
    
    # Show available tools
    from forge_ai.tools import get_all_tools
    tools = get_all_tools()
    
    print(f"\nThe AI has access to {len(tools)} tools:")
    for tool in tools[:5]:  # Show first 5
        print(f"  - {tool.name}: {tool.description[:60]}...")
    print(f"  ... and {len(tools) - 5} more")
    
    print()
    print("=" * 80)
    print("TOOL CALL FORMAT")
    print("=" * 80)
    print("""
When the AI needs to use a tool, it outputs:

<tool_call>
{"tool": "tool_name", "params": {"param1": "value1"}}
</tool_call>

The system executes the tool and injects the result:

<tool_result>
{"tool": "tool_name", "success": true, "result": "..."}
</tool_result>

The AI then continues based on the result.
""")
    
    print()
    print("=" * 80)
    print("EXAMPLE CONVERSATIONS")
    print("=" * 80)
    
    # Example 1: Image generation
    print("\n--- Example 1: Image Generation ---")
    print("User: Can you generate an image of a sunset?")
    print("AI: I'll create that image for you.")
    print("    <tool_call>")
    print('    {"tool": "generate_image", "params": {"prompt": "beautiful sunset", ...}}')
    print("    </tool_call>")
    print("    <tool_result>")
    print('    {"success": true, "result": "Image saved to outputs/sunset.png"}')
    print("    </tool_result>")
    print("    Done! I've created a beautiful sunset image at outputs/sunset.png.")
    
    # Example 2: Multi-step
    print("\n--- Example 2: Multi-step Tool Chain ---")
    print("User: Generate an image of a robot, then tell me what's in it")
    print("AI: I'll generate the robot image first.")
    print("    <tool_call>")
    print('    {"tool": "generate_image", "params": {"prompt": "friendly robot", ...}}')
    print("    </tool_call>")
    print("    <tool_result>")
    print('    {"success": true, "output_path": "outputs/robot.png"}')
    print("    </tool_result>")
    print("    Great! Now let me analyze what's in the image.")
    print("    <tool_call>")
    print('    {"tool": "analyze_image", "params": {"image_path": "outputs/robot.png"}}')
    print("    </tool_call>")
    print("    <tool_result>")
    print('    {"success": true, "result": "A friendly humanoid robot with blue eyes..."}')
    print("    </tool_result>")
    print("    The image shows a friendly humanoid robot with blue eyes!")
    
    print()
    print("=" * 80)
    print("TRAINING THE AI")
    print("=" * 80)
    print("""
To teach the AI to use tools:

1. Train the tokenizer with tool tokens:
   python -m forge_ai.core.tokenizer data/tool_training_data.txt

2. Train the model on tool examples:
   python run.py --train --data data/tool_training_data.txt

3. The AI learns to:
   - Recognize when tools are needed
   - Format tool calls correctly
   - Interpret results
   - Respond naturally
""")
    
    print()
    print("=" * 80)
    print("LIVE DEMO (if model is trained)")
    print("=" * 80)
    
    # Try a simple generation (will only work if model is trained)
    try:
        print("\nUser: Hello, how are you?")
        response = engine.generate(
            "Q: Hello, how are you?\nA:",
            max_gen=50,
            temperature=0.7,
            execute_tools=False  # Disable for simple greeting
        )
        print(f"AI: {response.split('A:')[-1].strip()[:100]}...")
    except Exception as e:
        print(f"Note: Model generation not available: {e}")
        print("This is expected if the model hasn't been trained yet.")
    
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Load modules for the tools you want to use:
   - manager.load('image_gen_local') for image generation
   - manager.load('vision') for image analysis
   - manager.load('avatar') for avatar control
   
2. Train your model with data/tool_training_data.txt

3. Enable tool execution and chat with the AI:
   response = engine.generate(prompt, execute_tools=True)

For more information, see docs/TOOL_USE.md
""")


if __name__ == "__main__":
    main()
