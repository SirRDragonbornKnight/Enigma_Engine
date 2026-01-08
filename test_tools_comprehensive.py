"""
Comprehensive Tool Testing Script for Enigma Engine
====================================================
Tests ALL tools and AI prompt-based tool execution (skipping robot, avatar, game)
"""

import json
import os
import sys
from pathlib import Path

# Add enigma to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_result(tool_name, result):
    """Pretty print a tool result."""
    print(f"\n{'='*60}")
    print(f"TOOL: {tool_name}")
    print(f"{'='*60}")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str)[:1500])  # Truncate long output
    else:
        print(str(result)[:1500])
    print()

def test_file_tools(registry):
    """Test all file-related tools."""
    print("\n\n" + "="*70)
    print("  FILE TOOLS - COMPREHENSIVE TEST")
    print("="*70)
    
    # Test list_directory
    result = registry.execute("list_directory", path=".")
    print_result("list_directory (current dir)", result)
    
    # Test read_file
    result = registry.execute("read_file", path="requirements.txt", max_lines=10)
    print_result("read_file (requirements.txt, 10 lines)", result)
    
    # Test write_file - create a test file
    test_content = "Hello from Enigma Engine test!\nLine 2\nLine 3"
    result = registry.execute("write_file", path="outputs/test_write.txt", content=test_content)
    print_result("write_file (create test file)", result)
    
    # Read it back
    result = registry.execute("read_file", path="outputs/test_write.txt")
    print_result("read_file (verify written content)", result)
    
    # Test move_file
    result = registry.execute("move_file", source="outputs/test_write.txt", destination="outputs/test_moved.txt")
    print_result("move_file (rename test file)", result)
    
    # Verify it moved
    result = registry.execute("list_directory", path="outputs")
    print_result("list_directory (verify move)", result)
    
    # Test delete_file
    result = registry.execute("delete_file", path="outputs/test_moved.txt")
    print_result("delete_file (cleanup test file)", result)
    
    return True

def test_system_tools(registry):
    """Test system tools."""
    print("\n\n" + "="*70)
    print("  SYSTEM TOOLS - COMPREHENSIVE TEST")
    print("="*70)
    
    # Get system info
    result = registry.execute("get_system_info")
    print_result("get_system_info", result)
    
    # Run commands
    result = registry.execute("run_command", command="echo Test from Enigma", timeout=5)
    print_result("run_command (echo)", result)
    
    result = registry.execute("run_command", command="dir /b", timeout=5)
    print_result("run_command (dir)", result)
    
    result = registry.execute("run_command", command="python -c \"print('Python works!')\"", timeout=5)
    print_result("run_command (python -c)", result)
    
    return True

def test_interactive_tools(registry):
    """Test interactive/task management tools."""
    print("\n\n" + "="*70)
    print("  INTERACTIVE TOOLS - COMPREHENSIVE TEST")
    print("="*70)
    
    from datetime import datetime, timedelta
    
    # Create checklist (now works with fixed param name!)
    result = registry.execute("create_checklist", name="AI Test Checklist", items=["Task A", "Task B", "Task C"])
    print_result("create_checklist", result)
    
    # List checklists
    result = registry.execute("list_checklists")
    print_result("list_checklists", result)
    
    # Add tasks with different priorities
    result = registry.execute("add_task", title="High priority task", priority="high")
    print_result("add_task (high priority)", result)
    
    result = registry.execute("add_task", title="Medium priority task", priority="medium", 
                              description="This is a test task")
    print_result("add_task (medium priority)", result)
    
    result = registry.execute("add_task", title="Low priority task", priority="low")
    print_result("add_task (low priority)", result)
    
    # List all tasks
    result = registry.execute("list_tasks")
    print_result("list_tasks (all)", result)
    
    # List by priority
    result = registry.execute("list_tasks", priority="high")
    print_result("list_tasks (high priority only)", result)
    
    # Set reminders
    future_time = (datetime.now() + timedelta(hours=2)).isoformat()
    result = registry.execute("set_reminder", message="Test reminder from comprehensive test", remind_at=future_time)
    print_result("set_reminder", result)
    
    # List reminders
    result = registry.execute("list_reminders")
    print_result("list_reminders", result)
    
    # Check due reminders
    result = registry.execute("check_reminders")
    print_result("check_reminders", result)
    
    return True

def test_web_tools(registry):
    """Test web tools."""
    print("\n\n" + "="*70)
    print("  WEB TOOLS - COMPREHENSIVE TEST")
    print("="*70)
    
    # Web search
    result = registry.execute("web_search", query="artificial intelligence", num_results=3)
    print_result("web_search (AI)", result)
    
    result = registry.execute("web_search", query="PyTorch deep learning", num_results=3)
    print_result("web_search (PyTorch)", result)
    
    # Fetch webpages
    result = registry.execute("fetch_webpage", url="https://httpbin.org/html")
    print_result("fetch_webpage (httpbin)", result)
    
    result = registry.execute("fetch_webpage", url="https://example.com")
    print_result("fetch_webpage (example.com)", result)
    
    return True

def test_document_tools(registry):
    """Test document tools."""
    print("\n\n" + "="*70)
    print("  DOCUMENT TOOLS - COMPREHENSIVE TEST")
    print("="*70)
    
    # Read various document types
    result = registry.execute("read_document", path="README.md")
    print_result("read_document (README.md)", result)
    
    result = registry.execute("read_document", path="CONTRIBUTING.md")
    print_result("read_document (CONTRIBUTING.md)", result)
    
    # Extract text
    result = registry.execute("extract_text", path="docs/TOOL_USE.md", max_chars=800)
    print_result("extract_text (TOOL_USE.md)", result)
    
    return True

def test_vision_tools(registry):
    """Test vision/screenshot tools."""
    print("\n\n" + "="*70)
    print("  VISION TOOLS - COMPREHENSIVE TEST")
    print("="*70)
    
    # Take screenshot
    result = registry.execute("screenshot", output_path="outputs/comprehensive_test_screenshot.png")
    print_result("screenshot (full screen)", result)
    
    # See screen (may require additional setup)
    result = registry.execute("see_screen")
    print_result("see_screen", result)
    
    # Find on screen
    result = registry.execute("find_on_screen", target="file")
    print_result("find_on_screen (looking for 'file')", result)
    
    return True

def test_tool_executor_with_ai_prompts():
    """Test the tool executor with AI-style prompts."""
    print("\n\n" + "="*70)
    print("  TOOL EXECUTOR - AI PROMPT SIMULATION")
    print("="*70)
    
    from enigma.tools.tool_executor import ToolExecutor
    
    executor = ToolExecutor()
    
    # Simulate AI outputs with tool calls
    ai_prompts = [
        # Prompt 1: File reading
        {
            "description": "AI wants to read a file",
            "ai_output": '''I'll read the requirements file for you.
<tool_call>
{"tool": "read_file", "params": {"path": "requirements.txt", "max_lines": 5}}
</tool_call>
Let me check what's in there.'''
        },
        # Prompt 2: Web search
        {
            "description": "AI wants to search the web",
            "ai_output": '''Let me search for information about Python.
<tool_call>
{"tool": "web_search", "params": {"query": "Python programming language", "num_results": 3}}
</tool_call>
I'll analyze the results.'''
        },
        # Prompt 3: System info
        {
            "description": "AI wants system information",
            "ai_output": '''I'll check your system specs.
<tool_call>
{"tool": "get_system_info", "params": {}}
</tool_call>
This will tell us about your hardware.'''
        },
        # Prompt 4: Run a command
        {
            "description": "AI wants to run a command",
            "ai_output": '''Let me check the Python version.
<tool_call>
{"tool": "run_command", "params": {"command": "python --version", "timeout": 10}}
</tool_call>
That should tell us.'''
        },
        # Prompt 5: Create a task
        {
            "description": "AI wants to create a task",
            "ai_output": '''I'll add that to your tasks.
<tool_call>
{"tool": "add_task", "params": {"title": "AI-created task", "description": "This task was created by the AI", "priority": "medium"}}
</tool_call>
Done!'''
        },
        # Prompt 6: List directory
        {
            "description": "AI wants to list files",
            "ai_output": '''Let me see what's in the docs folder.
<tool_call>
{"tool": "list_directory", "params": {"path": "docs"}}
</tool_call>
Here are the documentation files.'''
        },
        # Prompt 7: Multiple tool calls in one response
        {
            "description": "AI wants to do multiple things",
            "ai_output": '''I'll check both the system and search the web.
<tool_call>
{"tool": "get_system_info", "params": {}}
</tool_call>
And also:
<tool_call>
{"tool": "web_search", "params": {"query": "NVIDIA RTX 2080 specs", "num_results": 2}}
</tool_call>
Let me compile this info.'''
        },
        # Prompt 8: Fetch webpage
        {
            "description": "AI wants to fetch a webpage",
            "ai_output": '''Let me get the content from that page.
<tool_call>
{"tool": "fetch_webpage", "params": {"url": "https://httpbin.org/json"}}
</tool_call>
I'll summarize it for you.'''
        },
    ]
    
    for i, prompt_data in enumerate(ai_prompts, 1):
        print(f"\n{'-'*60}")
        print(f"AI PROMPT #{i}: {prompt_data['description']}")
        print(f"{'-'*60}")
        print(f"\nAI OUTPUT:\n{prompt_data['ai_output'][:300]}...")
        
        # Parse tool calls from AI output
        tool_calls = executor.parse_tool_calls(prompt_data['ai_output'])
        print(f"\nPARSED TOOL CALLS: {len(tool_calls)} found")
        
        for tool_name, params, start, end in tool_calls:
            print(f"  - Tool: {tool_name}")
            print(f"    Params: {params}")
            
            # Execute the tool
            result = executor.execute_tool(tool_name, params)
            
            print(f"\n  RESULT:")
            result_str = json.dumps(result, indent=4, default=str)
            # Truncate long results
            if len(result_str) > 800:
                result_str = result_str[:800] + "\n    ... (truncated)"
            print(f"    {result_str}")
    
    return True

def main():
    print("\n" + "="*70)
    print("  ENIGMA ENGINE - COMPREHENSIVE TOOL SYSTEM TEST")
    print("="*70)
    
    # Import the tool registry
    try:
        from enigma.tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        print("\n✓ Tool Registry loaded successfully!")
    except Exception as e:
        print(f"\n✗ Failed to load Tool Registry: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # List all available tools
    print("\n" + "-"*50)
    print("AVAILABLE TOOLS:")
    print("-"*50)
    tools = registry.list_tools()
    for tool in tools:
        print(f"  • {tool['name']}: {tool['description'][:50]}...")
    print(f"\nTotal: {len(tools)} tools available")
    
    # Run all tests
    tests = [
        ("File Tools", test_file_tools),
        ("System Tools", test_system_tools),
        ("Interactive Tools", test_interactive_tools),
        ("Web Tools", test_web_tools),
        ("Document Tools", test_document_tools),
        ("Vision Tools", test_vision_tools),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n\nRunning {test_name}...")
            success = test_func(registry)
            results[test_name] = "✓ PASSED" if success else "✗ FAILED"
        except Exception as e:
            results[test_name] = f"✗ ERROR: {str(e)[:50]}"
            import traceback
            traceback.print_exc()
    
    # Run AI prompt executor test
    try:
        print(f"\n\nRunning AI Prompt Executor Test...")
        success = test_tool_executor_with_ai_prompts()
        results["AI Prompt Executor"] = "✓ PASSED" if success else "✗ FAILED"
    except Exception as e:
        results["AI Prompt Executor"] = f"✗ ERROR: {str(e)[:50]}"
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    for test_name, status in results.items():
        print(f"  {test_name}: {status}")
    
    print("\n" + "="*70)
    print("  SKIPPED (as requested): robot, avatar, game")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
