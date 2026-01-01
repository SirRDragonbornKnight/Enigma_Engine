#!/usr/bin/env python3
"""
Demo: AI Tool Integration
==========================

This script demonstrates the new tool integration features:
1. Enhanced tokenizer with tool-specific special tokens
2. Tool interface for AI to invoke capabilities
3. Tool-aware generation that executes tools during inference

Run this to see how the AI can generate tool calls and have them executed.
"""

import sys
from pathlib import Path

# Add enigma to path
sys.path.insert(0, str(Path(__file__).parent))

from enigma.core.tool_interface import ToolInterface, parse_and_execute_tool
from enigma.core.tool_prompts import TOOL_SYSTEM_PROMPT, get_tool_enabled_system_prompt
from enigma.core.advanced_tokenizer import AdvancedBPETokenizer


def demo_tokenizer():
    """Demonstrate the upgraded tokenizer."""
    print("=" * 70)
    print("TOKENIZER DEMO")
    print("=" * 70)
    
    tokenizer = AdvancedBPETokenizer()
    
    print(f"\nTokenizer Statistics:")
    print(f"  Total vocab size: {tokenizer.vocab_size:,}")
    print(f"  Special tokens: {len(tokenizer.special_tokens)}")
    print(f"  Base tokens: {tokenizer.vocab_size - len(tokenizer.special_tokens):,}")
    
    print(f"\nSpecial Tokens for Tool Use:")
    tool_tokens = [
        '<|tool_call|>', '<|tool_result|>', '<|tool_end|>',
        '<|generate_image|>', '<|avatar_action|>', '<|speak|>',
        '<|vision|>', '<|search_web|>'
    ]
    for token in tool_tokens:
        if token in tokenizer.special_tokens:
            token_id = tokenizer.special_tokens[token]
            print(f"  {token:25s} -> ID {token_id}")
    
    print(f"\nTest Encoding/Decoding:")
    test_text = '<|tool_call|>generate_image("sunset")<|tool_end|>'
    print(f"  Input: {test_text}")
    ids = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"  Token IDs: {ids[:10]}..." if len(ids) > 10 else f"  Token IDs: {ids}")
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"  Decoded: {decoded}")


def demo_tool_interface():
    """Demonstrate the tool interface."""
    print("\n" + "=" * 70)
    print("TOOL INTERFACE DEMO")
    print("=" * 70)
    
    interface = ToolInterface()
    
    print(f"\nAvailable Tools: {len(interface.available_tools)}")
    for name, desc in interface.tool_descriptions.items():
        print(f"  - {name}: {desc[:60]}...")
    
    print(f"\nTest: Parse Tool Call")
    ai_output = '<|tool_call|>generate_image("a beautiful sunset")<|tool_end|>'
    print(f"  AI Output: {ai_output}")
    
    tool_call = interface.parse_tool_call(ai_output)
    if tool_call:
        print(f"  ✓ Parsed: {tool_call.tool_name}({tool_call.arguments})")
        
        result = interface.execute_tool(tool_call)
        print(f"  ✓ Executed: success={result.success}")
        
        formatted = interface.format_tool_result(result)
        print(f"  ✓ Result: {formatted[:80]}...")
    else:
        print(f"  ✗ Failed to parse tool call")
    
    print(f"\nTest: Multiple Tools")
    test_cases = [
        '<|tool_call|>speak("Hello world")<|tool_end|>',
        '<|tool_call|>search_web("latest AI news")<|tool_end|>',
        '<|tool_call|>avatar_action("set_expression", {"expression": "happy"})<|tool_end|>',
    ]
    
    for test_ai_output in test_cases:
        tool_call = interface.parse_tool_call(test_ai_output)
        if tool_call:
            print(f"  ✓ {tool_call.tool_name}")


def demo_system_prompt():
    """Demonstrate the tool system prompt."""
    print("\n" + "=" * 70)
    print("TOOL SYSTEM PROMPT DEMO")
    print("=" * 70)
    
    prompt = get_tool_enabled_system_prompt()
    
    # Show first part of prompt
    lines = prompt.split('\n')
    print(f"\nSystem Prompt Preview ({len(lines)} lines):")
    for line in lines[:15]:
        print(f"  {line}")
    print(f"  ... ({len(lines) - 15} more lines)")


def demo_training_data():
    """Show training data examples."""
    print("\n" + "=" * 70)
    print("TRAINING DATA DEMO")
    print("=" * 70)
    
    data_file = Path(__file__).parent / "data" / "tool_training_examples.txt"
    
    if data_file.exists():
        print(f"\nTraining data file: {data_file}")
        
        with open(data_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        total_lines = len(lines)
        
        # Count examples
        examples = content.count('Q:')
        
        print(f"  Total lines: {total_lines:,}")
        print(f"  Training examples: {examples}")
        
        # Show first example
        print(f"\nFirst Training Example:")
        in_example = False
        line_count = 0
        for line in lines:
            if line.startswith('Q:'):
                in_example = True
            if in_example:
                print(f"  {line}")
                line_count += 1
                if line_count > 10:
                    break
    else:
        print(f"\nTraining data file not found: {data_file}")


def demo_integration():
    """Demonstrate end-to-end integration."""
    print("\n" + "=" * 70)
    print("END-TO-END INTEGRATION DEMO")
    print("=" * 70)
    
    print("\nSimulating AI Generation with Tool Use:")
    
    # Simulate what would happen during generation
    interface = ToolInterface()
    
    scenarios = [
        {
            "user_query": "Can you show me a dragon?",
            "ai_output": 'I\'ll generate a dragon image for you!\n<|tool_call|>generate_image("a majestic dragon with iridescent scales")<|tool_end|>',
        },
        {
            "user_query": "Tell me about AI",
            "ai_output": 'Let me search for the latest information.\n<|tool_call|>search_web("artificial intelligence latest developments")<|tool_end|>',
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        print(f"User: {scenario['user_query']}")
        print(f"AI: {scenario['ai_output'][:50]}...")
        
        # Parse and execute
        result = parse_and_execute_tool(scenario['ai_output'], module_manager=None)
        if result:
            print(f"Tool Result: {result[:80]}...")
            print("✓ Tool executed successfully!")
        else:
            print("✗ No tool call detected")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("ENIGMA AI TOOL INTEGRATION DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows the new AI tool integration features:")
    print("1. Upgraded tokenizer with 39 special tokens for tool use")
    print("2. Tool interface supporting 8+ tools")
    print("3. Training data with 65+ examples")
    print("4. Integration with inference engine")
    
    try:
        demo_tokenizer()
        demo_tool_interface()
        demo_system_prompt()
        demo_training_data()
        demo_integration()
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Train your AI model with data/tool_training_examples.txt")
        print("2. Use generate_with_tools() for tool-aware generation")
        print("3. Enable tools by loading appropriate modules")
        print("\nFor more info, see:")
        print("  - enigma/core/tool_interface.py")
        print("  - enigma/core/tool_prompts.py")
        print("  - enigma/core/inference.py (generate_with_tools)")
        print()
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
