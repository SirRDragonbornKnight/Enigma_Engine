# Module System Improvements - Part 2

## Summary

This PR adds the remaining module system improvements that weren't covered in the first PR, completing the comprehensive module management system for Enigma Engine.

## Features Implemented

### 1. Module Health Checks
- **ModuleHealth dataclass**: Tracks health status, response time, error count, and warnings
- **health_check(module_id)**: Check health of individual modules
- **health_check_all()**: Check health of all loaded modules
- **start_health_monitor()**: Background monitoring with configurable intervals
- **stop_health_monitor()**: Stop background monitoring
- **is_health_monitor_running()**: Public API to check monitor status

### 2. Module Sandboxing
- **SandboxConfig**: Configuration for sandbox restrictions
- **ModuleSandbox**: Isolated execution environment with:
  - File system access restrictions (path whitelisting)
  - Network access control (allow/deny lists)
  - Memory and CPU limits (Unix-like systems)
  - Import restrictions (block dangerous modules)
- **load_sandboxed()**: Load modules in sandboxed environment

### 3. Module Documentation Generator
- **ModuleDocGenerator**: Automatic documentation from metadata
- **generate_markdown()**: Generate markdown for single or all modules
- **generate_html()**: Generate HTML documentation
- **generate_dependency_graph()**: Create dependency graphs in Mermaid or Graphviz format
- **export_to_file()**: Save documentation to files

### 4. Module Update Mechanism
- **ModuleUpdate dataclass**: Information about available updates
- **ModuleUpdater**: Version management system with:
  - **check_updates()**: Check for available updates
  - **get_changelog()**: Retrieve version changelogs
  - **update_module()**: Update with backup/restore
  - **rollback()**: Revert to previous version
  - **set_auto_update()**: Configure automatic updates
  - **get_update_status()**: Get current update settings

## Testing

All functionality is thoroughly tested with 41 comprehensive tests:
- 6 health check tests
- 9 sandboxing tests
- 10 documentation generation tests
- 13 update mechanism tests
- 3 integration tests

**All 41 tests passing!**

## Usage Examples

### Health Checks
```python
from enigma.modules import ModuleManager
from enigma.modules.registry import register_all

manager = ModuleManager()
register_all(manager)
manager.load('memory')

# Check health
health = manager.health_check('memory')
print(f"Healthy: {health.is_healthy}")
print(f"Response time: {health.response_time_ms}ms")

# Start background monitoring
manager.start_health_monitor(interval_seconds=60)
```

### Sandboxing
```python
from enigma.modules import SandboxConfig

# Create sandbox configuration
config = SandboxConfig(
    max_memory_mb=512,
    allow_network=False,
    allow_subprocess=False,
    restricted_imports=['os.system', 'subprocess']
)

# Load module in sandbox
manager.load_sandboxed('untrusted_module', config)
```

### Documentation Generation
```python
from enigma.modules import ModuleDocGenerator

doc_gen = ModuleDocGenerator(manager)

# Generate markdown for one module
markdown = doc_gen.generate_markdown('model')

# Generate dependency graph
graph = doc_gen.generate_dependency_graph('mermaid')

# Export all documentation
doc_gen.export_to_file('docs/modules.md', 'markdown')
```

### Update Checking
```python
from enigma.modules import ModuleUpdater

updater = ModuleUpdater(manager)

# Check for updates
updates = updater.check_updates()
for update in updates:
    print(f"{update.module_id}: {update.current_version} -> {update.latest_version}")

# Enable auto-update
updater.set_auto_update(True, check_interval_hours=24)
```

## Design Considerations

1. **Health Monitoring**: Background thread with configurable intervals to avoid blocking
2. **Sandboxing**: Basic isolation suitable for development; production should use containers
3. **Documentation**: Auto-generated from existing metadata, stays in sync with code
4. **Updates**: Stub implementation ready for production integration with update registry

## Security

- CodeQL security scan: **0 alerts**
- Sandbox provides basic isolation for untrusted modules
- Import restrictions block dangerous operations
- Resource limits prevent DoS attacks

## Compatibility

- Python 3.9+
- All existing tests continue to pass
- No breaking changes to existing API
- New functionality is opt-in

## Future Enhancements

1. **Semantic Versioning**: Use `packaging.version.parse()` for proper version comparison
2. **Production Update Registry**: Integrate with actual update server
3. **Enhanced Sandboxing**: Add more fine-grained permission controls
4. **Metrics Collection**: Add performance metrics to health checks
5. **Alert System**: Add notifications for unhealthy modules

## Files Changed

- `enigma/modules/manager.py`: Added health check methods and sandboxed loading
- `enigma/modules/sandbox.py`: New file for sandboxing functionality
- `enigma/modules/docs.py`: New file for documentation generation
- `enigma/modules/updater.py`: New file for update management
- `enigma/modules/__init__.py`: Updated exports
- `tests/test_modules_extended.py`: New comprehensive test suite
- `demo_module_improvements.py`: Demo script showing all features

## Acceptance Criteria

✅ `manager.health_check('model')` returns ModuleHealth with status  
✅ `manager.start_health_monitor(60)` starts background monitoring  
✅ `ModuleSandbox` restricts file/network access as configured  
✅ `manager.load_sandboxed('untrusted_module', config)` loads with restrictions  
✅ `ModuleDocGenerator.generate_markdown('model')` produces valid markdown  
✅ `ModuleDocGenerator.generate_dependency_graph()` outputs mermaid diagram  
✅ `ModuleUpdater.check_updates()` returns list of available updates  
✅ All new code has docstrings and type hints  
✅ All existing tests continue to pass  
✅ 41 new tests all passing  
✅ Code review feedback addressed  
✅ Security scan clean (0 alerts)
