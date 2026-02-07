#!/usr/bin/env python3
"""Comprehensive module check - verifies all modules work."""

import sys
sys.path.insert(0, '.')

from enigma_engine.modules.registry import MODULE_REGISTRY
from enigma_engine.modules.manager import ModuleManager

print("=" * 60)
print("FORGE AI - COMPREHENSIVE MODULE CHECK")
print("=" * 60)

mgr = ModuleManager()
results = {'ok': [], 'warn': [], 'fail': []}

for mid, mcls in sorted(MODULE_REGISTRY.items()):
    try:
        m = mcls(manager=mgr)
        r = m.load()
        if r:
            results['ok'].append(mid)
            print(f"  [OK] {mid}")
        else:
            results['warn'].append(mid)
            print(f"  [WARN] {mid} - load() returned False")
    except Exception as e:
        err = str(e)[:60].replace('\n', ' ')
        results['fail'].append((mid, err))
        print(f"  [FAIL] {mid}: {err}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total modules: {len(MODULE_REGISTRY)}")
print(f"  OK:   {len(results['ok'])}")
print(f"  WARN: {len(results['warn'])}")
print(f"  FAIL: {len(results['fail'])}")

if results['fail']:
    print("\nFailed modules need attention:")
    for mid, err in results['fail']:
        print(f"  - {mid}: {err}")

# Check for missing tabs
print("\n" + "=" * 60)
print("CHECKING TAB FILES")
print("=" * 60)

import os
from pathlib import Path

tabs_dir = Path("enigma_engine/gui/tabs")
tab_files = [f.stem for f in tabs_dir.glob("*.py") if not f.name.startswith("_")]
print(f"Found {len(tab_files)} tab files")

# Check which modules have corresponding tabs
module_tab_mapping = {
    'camera': 'camera_tab',
    'gif_gen': 'gif_tab',
    'voice_clone': 'voice_clone_tab',
    'notes': 'notes_tab',
    'sessions': 'sessions_tab',
    'scheduler': 'scheduler_tab',
    'personality': 'personality_tab',
    'terminal': 'terminal_tab',
    'analytics': 'analytics_tab',
    'dashboard': 'dashboard_tab',
    'examples': 'examples_tab',
    'instructions': 'instructions_tab',
    'logs': 'logs_tab',
    'model_router': 'model_router_tab',
    'scaling': 'scaling_tab',
    'workspace': 'workspace_tab',
    'game_ai': None,  # Uses settings_tab game section
    'robot_control': None,  # Uses settings_tab robot section
    'huggingface': 'huggingface_providers',  # Part of chat providers
}

missing_tabs = []
for mod_id, expected_tab in module_tab_mapping.items():
    if expected_tab and expected_tab not in tab_files:
        missing_tabs.append((mod_id, expected_tab))
        print(f"  [MISSING] {mod_id} -> {expected_tab}.py")
    elif expected_tab:
        print(f"  [OK] {mod_id} -> {expected_tab}.py")
    else:
        print(f"  [SKIP] {mod_id} (integrated elsewhere)")

# Check tool definitions
print("\n" + "=" * 60)
print("CHECKING TOOL DEFINITIONS")
print("=" * 60)

try:
    from enigma_engine.tools.tool_definitions import get_all_tools
    tools = get_all_tools()
    print(f"Found {len(tools)} tool definitions")
    for tool in tools[:10]:
        print(f"  - {tool.name}")
    if len(tools) > 10:
        print(f"  ... and {len(tools) - 10} more")
except Exception as e:
    print(f"  [ERROR] Could not load tools: {e}")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)
