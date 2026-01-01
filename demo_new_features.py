#!/usr/bin/env python3
"""
Demo of text formatting features.

Shows how the AI can use formatting to express emphasis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from enigma.utils.text_formatting import TextFormatter
from enigma.utils.system_messages import system_msg, info_msg, warning_msg, error_msg

def demo_formatting():
    """Demonstrate text formatting capabilities."""
    print("=" * 70)
    print("TEXT FORMATTING DEMO")
    print("=" * 70)
    print()
    
    examples = [
        "This is **bold** text!",
        "This is *italic* for emphasis.",
        "This is __underlined__ because it's important!",
        "Use `print()` to output in Python.",
        "This is ~~incorrect~~ actually correct.",
        "# This is a Big Announcement!",
        "## This is a smaller heading",
        "> This is a quote from someone wise",
    ]
    
    for example in examples:
        print(f"Input:  {example}")
        html = TextFormatter.to_html(example)
        print(f"HTML:   {html}")
        print()

def demo_system_messages():
    """Demonstrate system message formatting."""
    print("=" * 70)
    print("SYSTEM MESSAGE DEMO")
    print("=" * 70)
    print()
    
    print(system_msg("AI Engine initialized successfully"))
    print(info_msg("Model loaded: enigma-small.pth"))
    print(info_msg("Vocabulary size: 8000 tokens"))
    print(warning_msg("Model appears untrained"))
    print(error_msg("Failed to load tokenizer"))
    print()

def demo_combined():
    """Demonstrate combined usage in a chat scenario."""
    print("=" * 70)
    print("COMBINED USAGE DEMO (Simulated Chat)")
    print("=" * 70)
    print()
    
    # User message
    print("User: Tell me about Python")
    print()
    
    # AI response with formatting
    ai_response = """Python is a **high-level programming language** that's known for its 
*simplicity* and __readability__. Here's what makes it great:

# Key Features
- Easy to learn syntax
- Powerful standard library
- Great for beginners and experts

To print text, use: `print("Hello, World!")`

> "Python is the second best language for everything" - Guido van Rossum
"""
    
    print("AI (raw):")
    print(ai_response)
    print()
    
    print("AI (formatted as HTML):")
    print(TextFormatter.to_html(ai_response))
    print()

if __name__ == "__main__":
    demo_formatting()
    print("\n")
    demo_system_messages()
    print("\n")
    demo_combined()
