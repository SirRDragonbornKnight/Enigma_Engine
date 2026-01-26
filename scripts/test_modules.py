#!/usr/bin/env python3
"""Test all module load() methods."""

import sys
sys.path.insert(0, '.')

from forge_ai.modules.registry import MODULE_REGISTRY
from forge_ai.modules.manager import ModuleManager

modules_to_test = [
    'camera', 'gif_gen', 'voice_clone', 'notes', 'sessions',
    'scheduler', 'personality', 'terminal', 'analytics',
    'dashboard', 'examples', 'instructions', 'logs',
    'model_router', 'scaling', 'game_ai', 'robot_control', 'workspace', 'huggingface'
]

print('Testing module load()...')
passed = 0
failed = 0

# Create a module manager for testing
manager = ModuleManager()

for mod_id in modules_to_test:
    if mod_id in MODULE_REGISTRY:
        mod_class = MODULE_REGISTRY[mod_id]
        try:
            instance = mod_class(manager=manager)
            result = instance.load()
            status = 'OK' if result else 'FAILED'
            if result:
                passed += 1
            else:
                failed += 1
            print(f'  {mod_id}: {status}')
        except Exception as e:
            failed += 1
            print(f'  {mod_id}: ERROR - {e}')
    else:
        failed += 1
        print(f'  {mod_id}: NOT FOUND')

print(f'\nResults: {passed} passed, {failed} failed')
