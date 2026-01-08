"""
Simple Tool Testing Script for Enigma Engine
=============================================
Tests various tools (skipping robot, avatar, game as requested)
"""

import json
import os
import sys

# Add enigma to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_result(tool_name, result):
    """Pretty print a tool result."""
    print(f"\n{'='*60}")
    print(f"TOOL: {tool_name}")
    print(f"{'='*60}")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str)[:1000])  # Truncate long output
    else:
        print(str(result)[:1000])
    print()

def main():
    print("\n" + "="*70)
    print("  ENIGMA ENGINE - TOOL SYSTEM TEST")
    print("="*70)
    
    # Import the tool registry
    try:
        from enigma.tools.tool_registry import ToolRegistry
        registry = ToolRegistry()
        print("\n✓ Tool Registry loaded successfully!")
    except Exception as e:
        print(f"\n✗ Failed to load Tool Registry: {e}")
        return
    
    # List all available tools
    print("\n" + "-"*50)
    print("AVAILABLE TOOLS:")
    print("-"*50)
    tools = registry.list_tools()
    for tool in tools:
        print(f"  • {tool['name']}: {tool['description'][:60]}...")
    
    # ==========================================
    # TEST 1: File Tools
    # ==========================================
    print("\n\n" + "="*70)
    print("  TEST 1: FILE TOOLS")
    print("="*70)
    
    # Test list_directory
    result = registry.execute("list_directory", path=".")
    print_result("list_directory (current dir)", result)
    
    # Test read_file (read this test script)
    result = registry.execute("read_file", path="requirements.txt", max_lines=10)
    print_result("read_file (requirements.txt, 10 lines)", result)
    
    # ==========================================
    # TEST 2: System Tools  
    # ==========================================
    print("\n\n" + "="*70)
    print("  TEST 2: SYSTEM TOOLS")
    print("="*70)
    
    # Test get_system_info
    result = registry.execute("get_system_info")
    print_result("get_system_info", result)
    
    # Test run_command (safe command)
    result = registry.execute("run_command", command="echo Hello from Enigma!", timeout=5)
    print_result("run_command (echo)", result)
    
    # Test run_command - get python version
    result = registry.execute("run_command", command="python --version", timeout=5)
    print_result("run_command (python --version)", result)
    
    # ==========================================
    # TEST 3: Interactive/Task Tools
    # ==========================================
    print("\n\n" + "="*70)
    print("  TEST 3: INTERACTIVE/TASK TOOLS")
    print("="*70)
    
    # Create a checklist - need 'name' and 'items' params
    # Note: 'name' conflicts with registry.execute's first param, so we rename it
    result = registry.execute("create_checklist", items=["Item 1", "Item 2", "Item 3"])
    print_result("create_checklist (items only, name is required param issue)", result)
    
    # Let's get the tool directly and call it
    checklist_tool = registry.get("create_checklist")
    if checklist_tool:
        result = checklist_tool.execute(name="Test Checklist", items=["Item 1", "Item 2", "Item 3"])
        print_result("create_checklist (direct call)", result)
    
    # Add a task - needs 'title' param (direct call)
    task_tool = registry.get("add_task")
    if task_tool:
        result = task_tool.execute(title="Test task item 1", priority="high")
        print_result("add_task", result)
    
    # List tasks (no params needed)
    result = registry.execute("list_tasks")
    print_result("list_tasks", result)
    
    # List all checklists
    result = registry.execute("list_checklists")
    print_result("list_checklists", result)
    
    # Set a reminder - needs message and remind_at in ISO format (direct call for 'message')
    from datetime import datetime, timedelta
    future_time = (datetime.now() + timedelta(hours=1)).isoformat()
    reminder_tool = registry.get("set_reminder")
    if reminder_tool:
        result = reminder_tool.execute(message="Remember to test more features", remind_at=future_time)
        print_result("set_reminder", result)
    
    # List reminders
    result = registry.execute("list_reminders")
    print_result("list_reminders", result)
    
    # ==========================================
    # TEST 4: Web Tools (may fail without internet)
    # ==========================================
    print("\n\n" + "="*70)
    print("  TEST 4: WEB TOOLS (requires internet)")
    print("="*70)
    
    # Test web_search
    result = registry.execute("web_search", query="Python programming", num_results=3)
    print_result("web_search", result)
    
    # Test fetch_webpage (simple page)
    result = registry.execute("fetch_webpage", url="https://httpbin.org/html")
    print_result("fetch_webpage (httpbin.org)", result)
    
    # ==========================================
    # TEST 5: Document Tools
    # ==========================================
    print("\n\n" + "="*70)
    print("  TEST 5: DOCUMENT TOOLS")
    print("="*70)
    
    # Test read_document on README
    result = registry.execute("read_document", path="README.md")
    print_result("read_document (README.md)", result)
    
    # Test extract_text
    result = registry.execute("extract_text", path="README.md", max_chars=500)
    print_result("extract_text (README.md, 500 chars)", result)
    
    # ==========================================
    # TEST 6: Vision Tools (screenshot)
    # ==========================================
    print("\n\n" + "="*70)
    print("  TEST 6: VISION TOOLS")
    print("="*70)
    
    # Test screenshot (may fail without display)
    try:
        result = registry.execute("screenshot", output_path="outputs/test_screenshot.png")
        print_result("screenshot", result)
    except Exception as e:
        print(f"Screenshot failed (expected if no display): {e}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n\n" + "="*70)
    print("  TEST COMPLETE!")
    print("="*70)
    print("\nTools tested:")
    print("  ✓ File Tools: list_directory, read_file")
    print("  ✓ System Tools: get_system_info, run_command")
    print("  ✓ Interactive Tools: create_checklist, add_task, list_tasks, set_reminder")
    print("  ✓ Web Tools: web_search, fetch_webpage")
    print("  ✓ Document Tools: read_document, extract_text")
    print("  ✓ Vision Tools: screenshot")
    print("\nSkipped (as requested): robot, avatar, game")
    print()

if __name__ == "__main__":
    main()
