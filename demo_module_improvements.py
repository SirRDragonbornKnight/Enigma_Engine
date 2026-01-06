#!/usr/bin/env python3
"""
Demo script showing the new module system improvements.

Tests:
- Health checks
- Sandboxing
- Documentation generation
- Update checking
"""

from enigma.modules import (
    ModuleManager,
    ModuleHealth,
    ModuleSandbox,
    SandboxConfig,
    ModuleDocGenerator,
    ModuleUpdater,
    ModuleUpdate
)
from enigma.modules.registry import register_all
from pathlib import Path
import tempfile

print("=" * 70)
print("Enigma Engine - Module System Improvements Demo")
print("=" * 70)
print()

# 1. Initialize system
print("1. Initializing ModuleManager...")
manager = ModuleManager()
register_all(manager)
print(f"   Registered {len(manager.module_classes)} modules")
print()

# 2. Health Checks
print("2. Testing Health Checks...")
# Load a module
manager.load('memory')
health = manager.health_check('memory')
if health:
    print(f"   Module: {health.module_id}")
    print(f"   Healthy: {health.is_healthy}")
    print(f"   Response time: {health.response_time_ms:.2f}ms")
    print(f"   Error count: {health.error_count}")
    print(f"   Warnings: {health.warnings if health.warnings else 'None'}")
else:
    print("   Health check returned None (module not loaded)")
print()

# 3. Sandboxing
print("3. Testing Sandboxing...")
config = SandboxConfig(
    max_memory_mb=512,
    allow_network=False,
    allow_subprocess=False
)
sandbox = ModuleSandbox('test', config)
print(f"   Created sandbox for module 'test'")
print(f"   Network allowed: {sandbox.check_permission('network')}")
print(f"   Subprocess allowed: {sandbox.check_permission('subprocess')}")

# Test running a function in sandbox
def test_func(x, y):
    return x * y

result = sandbox.run_in_sandbox(test_func, 5, 7)
print(f"   Executed function in sandbox: 5 * 7 = {result}")
print()

# 4. Documentation Generation
print("4. Testing Documentation Generation...")
doc_gen = ModuleDocGenerator(manager)

# Generate markdown for one module
markdown = doc_gen.generate_markdown('model')
print(f"   Generated markdown docs for 'model' ({len(markdown)} chars)")

# Generate dependency graph
graph = doc_gen.generate_dependency_graph('mermaid')
print(f"   Generated Mermaid dependency graph ({len(graph)} chars)")

# Export to file
with tempfile.TemporaryDirectory() as tmpdir:
    doc_path = Path(tmpdir) / 'module_docs.md'
    doc_gen.export_to_file(doc_path, 'markdown')
    print(f"   Exported documentation to {doc_path.name}")
    print(f"   File size: {doc_path.stat().st_size} bytes")
print()

# 5. Update Mechanism
print("5. Testing Update Mechanism...")
updater = ModuleUpdater(manager)
print(f"   Registry URL: {updater.registry_url}")
print(f"   Backup directory: {updater.backup_dir}")

# Check for updates
updates = updater.check_updates()
print(f"   Checked for updates: {len(updates)} available")

# Get update status
status = updater.get_update_status()
print(f"   Auto-update enabled: {status['auto_update_enabled']}")
print(f"   Check interval: {status['check_interval_hours']} hours")
print()

# 6. Health Monitoring
print("6. Testing Background Health Monitoring...")
manager.start_health_monitor(interval_seconds=5)
print("   Started health monitor (5s interval)")
import time
print("   Waiting 2 seconds...")
time.sleep(2)
manager.stop_health_monitor()
print("   Stopped health monitor")
print()

print("=" * 70)
print("All tests completed successfully!")
print("=" * 70)
