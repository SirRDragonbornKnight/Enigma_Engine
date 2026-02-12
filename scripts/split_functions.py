#!/usr/bin/env python3
"""
Split Large Functions Script - Enigma AI Engine

This script refactors large GUI functions by extracting sections into helper functions.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

ROOT = Path(__file__).parent.parent
ENGINE_DIR = ROOT / "enigma_engine"


def split_settings_tab():
    """Split the massive create_settings_tab function into smaller helpers."""
    filepath = ENGINE_DIR / "gui" / "tabs" / "settings_tab.py"
    content = filepath.read_text(encoding='utf-8')
    lines = content.splitlines(keepends=True)
    
    # Find the function and its sections
    func_start = None
    func_end = None
    sections = []
    
    for i, line in enumerate(lines):
        if 'def create_settings_tab(parent):' in line:
            func_start = i
        elif func_start is not None and func_end is None:
            # Check for next function at same indent level
            if re.match(r'^def \w+\(', line) and 'create_settings_tab' not in line:
                func_end = i
                break
            # Track section markers
            if '# ===' in line and '====' in line:
                # Extract section name
                match = re.search(r'# === (.+?) ===', line)
                if match:
                    section_name = match.group(1).strip()
                    sections.append((i, section_name))
    
    if not func_start or not sections:
        print("Could not find function or sections")
        return
    
    print(f"Found create_settings_tab at line {func_start + 1}")
    print(f"Found {len(sections)} sections to extract")
    
    # Calculate where each section ends (at the next section or function end)
    section_ranges = []
    for i, (lineno, name) in enumerate(sections):
        start = lineno
        if i + 1 < len(sections):
            end = sections[i + 1][0]
        else:
            end = func_end if func_end else len(lines)
        section_ranges.append((start, end, name))
    
    # Create helper function names
    def make_func_name(section_name: str) -> str:
        name = section_name.lower()
        name = re.sub(r'[^a-z0-9]+', '_', name)
        name = name.strip('_')
        return f"_create_{name}_section"
    
    # Generate the helper functions
    helper_funcs = []
    call_statements = []
    
    for start, end, name in section_ranges:
        func_name = make_func_name(name)
        
        # Extract section code (indent is 4 spaces in original)
        section_lines = lines[start:end]
        
        # Remove leading section comment and adjust to function
        # First line is the section comment
        code_lines = section_lines[1:]  # Skip the === comment
        
        # Dedent by 4 spaces (was in the main function)
        dedented = []
        for line in code_lines:
            if line.startswith("    "):
                dedented.append(line[4:])  # Remove one indent level
            else:
                dedented.append(line)
        
        # Create helper function
        helper = f"""
def {func_name}(parent, layout):
    \"\"\"{name} section.\"\"\"
{''.join(dedented)}
"""
        helper_funcs.append((func_name, helper, name))
        call_statements.append(f"    {func_name}(parent, layout)")
    
    # Now rewrite the file with helper functions and refactored main function
    print(f"\nGenerated {len(helper_funcs)} helper functions")
    for name, _, section in helper_funcs[:10]:
        print(f"  {name}() - {section}")
    if len(helper_funcs) > 10:
        print(f"  ... and {len(helper_funcs) - 10} more")


def analyze_large_functions():
    """Analyze all large functions and suggest splits."""
    import ast
    
    large_files = [
        ("enigma_engine/gui/tabs/settings_tab.py", "create_settings_tab"),
        ("enigma_engine/gui/tabs/avatar/avatar_display.py", "create_avatar_subtab"),
        ("enigma_engine/gui/tabs/training_tab.py", "create_training_tab"),
        ("enigma_engine/gui/enhanced_window.py", "_build_ui"),
    ]
    
    for filepath, func_name in large_files:
        path = ROOT / filepath
        if not path.exists():
            continue
            
        content = path.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        # Find section markers
        sections = []
        in_func = False
        
        for i, line in enumerate(lines):
            if f'def {func_name}(' in line:
                in_func = True
            elif in_func:
                if re.match(r'^def \w+\(', line.strip()):
                    if func_name not in line:
                        break
                # Look for section markers (# === or # ----)
                if re.search(r'#\s*(===|----).*(===|----)', line):
                    sections.append((i + 1, line.strip()))
        
        print(f"\n{filepath} - {func_name}(): {len(sections)} sections")
        for lineno, marker in sections[:10]:
            print(f"  Line {lineno}: {marker[:60]}...")


def main():
    os.chdir(ROOT)
    
    print("=" * 60)
    print("Large Function Analysis")
    print("=" * 60)
    
    analyze_large_functions()
    
    print("\n" + "=" * 60)
    print("Settings Tab Split Preview")
    print("=" * 60)
    
    split_settings_tab()


if __name__ == "__main__":
    main()
