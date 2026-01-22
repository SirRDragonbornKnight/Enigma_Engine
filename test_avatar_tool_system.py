#!/usr/bin/env python3
"""
Background Processing Visualizer & Tester
=========================================
Tests tool-based avatar control and shows what the AI processes in the background.

Usage:
    python test_avatar_tool_system.py
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forge_ai.tools.tool_executor import ToolExecutor
from forge_ai.modules import ModuleManager


def visualize_tool_call(text: str):
    """Show what the AI sees vs what the user sees."""
    print("\n" + "="*80)
    print("BACKGROUND PROCESSING VISUALIZATION")
    print("="*80)
    
    # Split visible text from tool calls
    lines = text.split('\n')
    visible_text = []
    background_calls = []
    
    in_tool_call = False
    in_tool_result = False
    current_tool = ""
    
    for line in lines:
        if '<tool_call>' in line:
            in_tool_call = True
            current_tool = ""
        elif '</tool_call>' in line:
            in_tool_call = False
            background_calls.append(("TOOL CALL", current_tool))
            current_tool = ""
        elif '<tool_result>' in line:
            in_tool_result = True
            current_tool = ""
        elif '</tool_result>' in line:
            in_tool_result = False
            background_calls.append(("TOOL RESULT", current_tool))
            current_tool = ""
        elif in_tool_call or in_tool_result:
            current_tool += line
        else:
            if line.strip() and not line.startswith('Q:') and not line.startswith('A:'):
                visible_text.append(line.strip())
    
    # Show what user sees
    print("\n🔵 USER SEES (visible text + TTS):")
    print("-" * 80)
    for line in visible_text:
        if line:
            print(f"  💬 {line}")
    
    # Show background processing
    print("\n🔴 BACKGROUND PROCESSING (invisible to user):")
    print("-" * 80)
    for call_type, data in background_calls:
        try:
            parsed = json.loads(data.strip())
            if call_type == "TOOL CALL":
                print(f"  ⚙️  EXECUTING: {parsed.get('tool', 'unknown')}")
                print(f"      └─ Parameters: {json.dumps(parsed.get('params', {}), indent=8)}")
            else:  # TOOL RESULT
                success = "✅ SUCCESS" if parsed.get('success') else "❌ FAILED"
                print(f"  📥 RESULT: {success}")
                if 'result' in parsed:
                    print(f"      └─ {parsed['result']}")
        except:
            print(f"  {call_type}: {data[:60]}...")
    
    print()


def test_tool_execution():
    """Test actual tool execution."""
    print("\n" + "="*80)
    print("LIVE TOOL EXECUTION TEST")
    print("="*80)
    
    try:
        manager = ModuleManager()
        executor = ToolExecutor(manager)
        
        # Test avatar control tool
        test_command = '''Q: wave hello
A: Hello! 👋
<tool_call>{"tool": "control_avatar_bones", "params": {"action": "gesture", "gesture_name": "wave"}}</tool_call>'''
        
        print("\n🧪 Testing: 'wave hello'")
        print("-" * 80)
        
        # Parse tool calls
        tool_calls = executor.parse_tool_calls(test_command)
        
        if not tool_calls:
            print("❌ No tool calls detected")
            return
        
        print(f"✅ Detected {len(tool_calls)} tool call(s)")
        
        for i, (tool_name, params, start, end) in enumerate(tool_calls, 1):
            print(f"\n🔧 Tool Call #{i}:")
            print(f"   Tool: {tool_name}")
            print(f"   Params: {json.dumps(params, indent=10)}")
            
            # Execute the tool
            result = executor.execute(tool_name, params)
            
            print(f"\n📊 Result:")
            print(f"   Success: {result.get('success', False)}")
            if 'result' in result:
                print(f"   Message: {result['result']}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


def show_comparison():
    """Show side-by-side comparison of old vs new approach."""
    print("\n" + "="*80)
    print("COMPARISON: Tag-Based vs Tool-Based")
    print("="*80)
    
    print("\n📜 OLD APPROACH (Tag-Based):")
    print("-" * 80)
    print("User says: 'wave hello'")
    print()
    print("AI responds:")
    print("  Hello! <bone_control>right_arm|pitch=45,yaw=30,roll=0</bone_control>")
    print()
    print("❌ PROBLEMS:")
    print("  • User sees ugly tags in chat")
    print("  • TTS reads: 'Hello bone control right arm pitch equals forty five...'")
    print("  • Hard to read")
    print("  • Cluttered interface")
    
    print("\n✨ NEW APPROACH (Tool-Based):")
    print("-" * 80)
    print("User says: 'wave hello'")
    print()
    print("AI responds:")
    print("  Hello! 👋")
    print()
    print("Background processing:")
    print("  [Silently executes: control_avatar_bones(action='gesture', gesture_name='wave')]")
    print()
    print("✅ BENEFITS:")
    print("  • User sees clean text: 'Hello! 👋'")
    print("  • TTS reads: 'Hello!' (natural)")
    print("  • Avatar waves simultaneously")
    print("  • Professional UX")


def load_training_examples():
    """Load and visualize training data."""
    print("\n" + "="*80)
    print("TRAINING DATA VISUALIZATION")
    print("="*80)
    
    training_file = Path(__file__).parent / "data/specialized/avatar_control_training_tool_based.txt"
    
    if not training_file.exists():
        print(f"❌ Training file not found: {training_file}")
        return
    
    with open(training_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find first complete example
    examples = content.split('Q: ')[1:4]  # Get first 3 examples
    
    for i, example in enumerate(examples, 1):
        print(f"\n📚 Example #{i}:")
        print("-" * 80)
        visualize_tool_call(f"Q: {example}")


def main():
    """Run all tests."""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "AVATAR TOOL SYSTEM TESTER" + " "*33 + "║")
    print("║" + " "*15 + "Background Processing Visualizer" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    
    # Show comparison
    show_comparison()
    
    # Visualize training examples
    load_training_examples()
    
    # Test live execution
    test_tool_execution()
    
    print("\n" + "="*80)
    print("✅ SUMMARY")
    print("="*80)
    print("""
The AI processes TWO layers simultaneously:

1. FOREGROUND (User sees):
   - Natural conversation text
   - Emojis and expressions
   - Clean, readable responses
   - TTS reads only this

2. BACKGROUND (Invisible):
   - Tool function calls
   - Parameter passing
   - Avatar bone manipulation
   - Error handling
   - Result logging

This separation creates a professional experience where the avatar moves
naturally during conversation without cluttering the chat interface.
""")
    
    print("🎯 Next Steps:")
    print("  1. Train model: python scripts/train_avatar_control.py --tool-based")
    print("  2. Test in GUI: python run.py --gui")
    print("  3. Say 'wave hello' and watch avatar respond + text stays clean")
    print()


if __name__ == "__main__":
    main()
