"""
Extended tests for Module Manager system.

Tests for:
- Health checking
- Sandboxing
- Documentation generation
- Update mechanism
"""

import unittest
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma.modules import (
    ModuleManager, Module, ModuleInfo, ModuleState, 
    ModuleCategory, ModuleHealth
)
from enigma.modules.sandbox import (
    ModuleSandbox, SandboxConfig, SandboxViolationError,
    create_default_sandbox_config
)
from enigma.modules.docs import ModuleDocGenerator
from enigma.modules.updater import ModuleUpdater, ModuleUpdate
from enigma.modules.registry import register_all


class TestModuleHealthChecks(unittest.TestCase):
    """Test health check functionality."""
    
    def setUp(self):
        """Create a fresh module manager for each test."""
        self.manager = ModuleManager()
        register_all(self.manager)
    
    def test_health_check_unloaded_module(self):
        """Test health check on unloaded module returns None."""
        health = self.manager.health_check('nonexistent')
        self.assertIsNone(health)
    
    def test_health_check_loaded_module(self):
        """Test health check on loaded module."""
        # Load a simple module
        success = self.manager.load('memory')
        
        if success:  # Only test if module loaded successfully
            health = self.manager.health_check('memory')
            
            self.assertIsNotNone(health)
            self.assertIsInstance(health, ModuleHealth)
            self.assertEqual(health.module_id, 'memory')
            self.assertIsInstance(health.is_healthy, bool)
            self.assertIsInstance(health.response_time_ms, float)
            self.assertIsInstance(health.error_count, int)
            self.assertIsInstance(health.warnings, list)
            self.assertGreaterEqual(health.response_time_ms, 0)
    
    def test_health_check_all(self):
        """Test health check on all modules."""
        # Load some modules
        self.manager.load('memory')
        
        results = self.manager.health_check_all()
        
        self.assertIsInstance(results, dict)
        # Should have health status for loaded modules
        for module_id, health in results.items():
            self.assertIsInstance(health, ModuleHealth)
            self.assertEqual(health.module_id, module_id)
    
    def test_health_monitor_start_stop(self):
        """Test starting and stopping health monitor."""
        # Start monitor
        self.manager.start_health_monitor(interval_seconds=1)
        
        self.assertTrue(self.manager.is_health_monitor_running())
        
        # Wait a bit
        time.sleep(0.5)
        
        # Stop monitor
        self.manager.stop_health_monitor()
        
        self.assertFalse(self.manager.is_health_monitor_running())
    
    def test_health_monitor_already_running(self):
        """Test starting monitor when already running."""
        self.manager.start_health_monitor()
        
        # Try to start again - should handle gracefully
        self.manager.start_health_monitor()
        
        # Clean up
        self.manager.stop_health_monitor()
    
    def test_health_check_response_time(self):
        """Test that health check measures response time."""
        self.manager.load('memory')
        
        health = self.manager.health_check('memory')
        
        if health:
            # Response time should be measured
            self.assertGreater(health.response_time_ms, 0)
            # Should be reasonable (less than 1 second)
            self.assertLess(health.response_time_ms, 1000)


class TestModuleSandbox(unittest.TestCase):
    """Test module sandboxing."""
    
    def test_sandbox_config_creation(self):
        """Test creating sandbox configuration."""
        config = SandboxConfig(
            allowed_paths=[Path('/tmp')],
            max_memory_mb=512,
            allow_network=False
        )
        
        self.assertEqual(len(config.allowed_paths), 1)
        self.assertEqual(config.max_memory_mb, 512)
        self.assertFalse(config.allow_network)
    
    def test_create_default_sandbox_config(self):
        """Test creating default sandbox config."""
        config = create_default_sandbox_config('test_module')
        
        self.assertIsInstance(config, SandboxConfig)
        self.assertGreater(config.max_memory_mb, 0)
        self.assertGreater(config.max_cpu_seconds, 0)
        self.assertIsInstance(config.restricted_imports, list)
    
    def test_sandbox_initialization(self):
        """Test sandbox initialization."""
        config = SandboxConfig()
        sandbox = ModuleSandbox('test_module', config)
        
        self.assertEqual(sandbox.module_id, 'test_module')
        self.assertEqual(sandbox.config, config)
        self.assertIsInstance(sandbox.permissions, dict)
    
    def test_sandbox_check_permission(self):
        """Test checking sandbox permissions."""
        config = SandboxConfig(allow_network=True, allow_subprocess=False)
        sandbox = ModuleSandbox('test_module', config)
        
        self.assertTrue(sandbox.check_permission('network'))
        self.assertFalse(sandbox.check_permission('subprocess'))
    
    def test_sandbox_path_access(self):
        """Test sandbox path access checking."""
        temp_dir = Path(tempfile.gettempdir())
        config = SandboxConfig(allowed_paths=[temp_dir])
        sandbox = ModuleSandbox('test_module', config)
        
        # Should allow access to allowed path
        test_path = temp_dir / 'test.txt'
        self.assertTrue(sandbox.check_path_access(test_path))
        
        # Should deny access to non-allowed path
        other_path = Path('/etc/passwd')
        self.assertFalse(sandbox.check_path_access(other_path))
    
    def test_sandbox_host_access(self):
        """Test sandbox host access checking."""
        config = SandboxConfig(
            allow_network=True,
            allowed_hosts=['example.com'],
            blocked_hosts=['evil.com']
        )
        sandbox = ModuleSandbox('test_module', config)
        
        # Should allow listed host
        self.assertTrue(sandbox.check_host_access('example.com'))
        
        # Should block blocked host
        self.assertFalse(sandbox.check_host_access('evil.com'))
        
        # Should block unlisted host
        self.assertFalse(sandbox.check_host_access('other.com'))
    
    def test_sandbox_network_disabled(self):
        """Test sandbox with network disabled."""
        config = SandboxConfig(allow_network=False)
        sandbox = ModuleSandbox('test_module', config)
        
        # Should block all hosts when network disabled
        self.assertFalse(sandbox.check_host_access('example.com'))
    
    def test_sandbox_run_simple_function(self):
        """Test running a simple function in sandbox."""
        config = SandboxConfig()
        sandbox = ModuleSandbox('test_module', config)
        
        def simple_func(x, y):
            return x + y
        
        result = sandbox.run_in_sandbox(simple_func, 2, 3)
        self.assertEqual(result, 5)
    
    def test_sandbox_run_with_exception(self):
        """Test that sandbox propagates exceptions."""
        config = SandboxConfig()
        sandbox = ModuleSandbox('test_module', config)
        
        def failing_func():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            sandbox.run_in_sandbox(failing_func)


class TestModuleSandboxedLoading(unittest.TestCase):
    """Test loading modules with sandbox."""
    
    def setUp(self):
        """Create manager with registered modules."""
        self.manager = ModuleManager()
        register_all(self.manager)
    
    def test_load_sandboxed_basic(self):
        """Test loading a module in sandbox."""
        config = create_default_sandbox_config('memory')
        
        # Try to load memory module in sandbox
        success = self.manager.load_sandboxed('memory', config)
        
        if success:
            self.assertIn('memory', self.manager.modules)
            module = self.manager.get_module('memory')
            self.assertIsNotNone(module)
            # Module should have sandbox reference
            self.assertTrue(hasattr(module, '_sandbox'))
    
    def test_load_sandboxed_with_default_config(self):
        """Test loading with default sandbox config."""
        # Load without explicit config - should use defaults
        success = self.manager.load_sandboxed('memory')
        
        # Should work (or fail gracefully)
        self.assertIsInstance(success, bool)


class TestModuleDocGenerator(unittest.TestCase):
    """Test documentation generation."""
    
    def setUp(self):
        """Create manager and doc generator."""
        self.manager = ModuleManager()
        register_all(self.manager)
        self.doc_gen = ModuleDocGenerator(self.manager)
    
    def test_doc_generator_initialization(self):
        """Test doc generator initialization."""
        self.assertEqual(self.doc_gen.manager, self.manager)
    
    def test_generate_markdown_single_module(self):
        """Test generating markdown for a single module."""
        markdown = self.doc_gen.generate_markdown('model')
        
        self.assertIsInstance(markdown, str)
        self.assertIn('# ', markdown)  # Should have header
        self.assertIn('model', markdown.lower())  # Should mention module
        self.assertIn('Description', markdown)
        self.assertIn('Version', markdown)
    
    def test_generate_markdown_nonexistent_module(self):
        """Test generating docs for nonexistent module."""
        markdown = self.doc_gen.generate_markdown('nonexistent')
        
        self.assertIsInstance(markdown, str)
        self.assertIn('Error', markdown)
        self.assertIn('not found', markdown.lower())
    
    def test_generate_all_markdown(self):
        """Test generating markdown for all modules."""
        markdown = self.doc_gen.generate_all_markdown()
        
        self.assertIsInstance(markdown, str)
        self.assertIn('# Enigma Engine', markdown)
        self.assertIn('Module Documentation', markdown)
        self.assertIn('Table of Contents', markdown)
        
        # Should include multiple modules
        self.assertIn('model', markdown.lower())
        self.assertIn('tokenizer', markdown.lower())
    
    def test_generate_html(self):
        """Test generating HTML documentation."""
        html = self.doc_gen.generate_html('model')
        
        self.assertIsInstance(html, str)
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<html>', html)
        self.assertIn('</html>', html)
        self.assertIn('model', html.lower())
    
    def test_generate_dependency_graph_mermaid(self):
        """Test generating Mermaid dependency graph."""
        graph = self.doc_gen.generate_dependency_graph('mermaid')
        
        self.assertIsInstance(graph, str)
        self.assertIn('graph TD', graph)
        # Should have some module references
        self.assertGreater(len(graph), 50)
    
    def test_generate_dependency_graph_graphviz(self):
        """Test generating Graphviz dependency graph."""
        graph = self.doc_gen.generate_dependency_graph('graphviz')
        
        self.assertIsInstance(graph, str)
        self.assertIn('digraph ModuleDependencies', graph)
        # Should have some node definitions
        self.assertIn('[', graph)
        self.assertIn(']', graph)
    
    def test_generate_dependency_graph_invalid_format(self):
        """Test generating graph with invalid format."""
        with self.assertRaises(ValueError):
            self.doc_gen.generate_dependency_graph('invalid')
    
    def test_export_to_file_markdown(self):
        """Test exporting documentation to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'docs.md'
            
            self.doc_gen.export_to_file(output_path, 'markdown')
            
            self.assertTrue(output_path.exists())
            content = output_path.read_text()
            self.assertIn('Enigma Engine', content)
            self.assertGreater(len(content), 100)
    
    def test_export_to_file_mermaid(self):
        """Test exporting dependency graph to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'graph.mmd'
            
            self.doc_gen.export_to_file(output_path, 'mermaid')
            
            self.assertTrue(output_path.exists())
            content = output_path.read_text()
            self.assertIn('graph TD', content)


class TestModuleUpdater(unittest.TestCase):
    """Test module update mechanism."""
    
    def setUp(self):
        """Create manager and updater."""
        self.manager = ModuleManager()
        register_all(self.manager)
        self.updater = ModuleUpdater(self.manager)
    
    def test_updater_initialization(self):
        """Test updater initialization."""
        self.assertEqual(self.updater.manager, self.manager)
        self.assertIsInstance(self.updater.registry_url, str)
        self.assertTrue(self.updater.backup_dir.exists())
    
    def test_updater_custom_registry_url(self):
        """Test updater with custom registry URL."""
        custom_url = "https://custom.example.com/registry"
        updater = ModuleUpdater(self.manager, registry_url=custom_url)
        
        self.assertEqual(updater.registry_url, custom_url)
    
    def test_check_updates_all(self):
        """Test checking updates for all modules."""
        updates = self.updater.check_updates()
        
        # Should return a list (may be empty in stub implementation)
        self.assertIsInstance(updates, list)
        
        # If any updates, should be ModuleUpdate objects
        for update in updates:
            self.assertIsInstance(update, ModuleUpdate)
    
    def test_check_updates_single_module(self):
        """Test checking updates for a specific module."""
        updates = self.updater.check_updates('model')
        
        self.assertIsInstance(updates, list)
    
    def test_check_updates_nonexistent_module(self):
        """Test checking updates for nonexistent module."""
        updates = self.updater.check_updates('nonexistent')
        
        # Should handle gracefully
        self.assertIsInstance(updates, list)
    
    def test_get_changelog(self):
        """Test getting changelog between versions."""
        changelog = self.updater.get_changelog('model', '1.0.0', '2.0.0')
        
        self.assertIsInstance(changelog, str)
        self.assertGreater(len(changelog), 0)
    
    def test_update_module_nonexistent(self):
        """Test updating nonexistent module."""
        success = self.updater.update_module('nonexistent')
        
        self.assertFalse(success)
    
    def test_rollback_no_backup(self):
        """Test rollback when no backup exists."""
        # Create temporary updater with custom backup dir
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = ModuleUpdater(self.manager)
            updater.backup_dir = Path(tmpdir)
            
            success = updater.rollback('model')
            
            # Should fail gracefully when no backup
            self.assertFalse(success)
    
    def test_set_auto_update(self):
        """Test enabling/disabling auto-update."""
        # Enable auto-update
        self.updater.set_auto_update(True, check_interval_hours=12)
        
        self.assertTrue(self.updater._auto_update_enabled)
        self.assertEqual(self.updater._check_interval_hours, 12)
        
        # Disable auto-update
        self.updater.set_auto_update(False)
        
        self.assertFalse(self.updater._auto_update_enabled)
    
    def test_get_update_status(self):
        """Test getting update status."""
        status = self.updater.get_update_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('auto_update_enabled', status)
        self.assertIn('check_interval_hours', status)
        self.assertIn('registry_url', status)
        self.assertIn('backup_dir', status)
        self.assertIn('available_backups', status)
    
    def test_module_update_dataclass(self):
        """Test ModuleUpdate dataclass."""
        update = ModuleUpdate(
            module_id='test',
            current_version='1.0.0',
            latest_version='2.0.0',
            changelog='Test changes',
            is_breaking=True
        )
        
        self.assertEqual(update.module_id, 'test')
        self.assertEqual(update.current_version, '1.0.0')
        self.assertEqual(update.latest_version, '2.0.0')
        self.assertTrue(update.is_breaking)


class TestIntegrationExtended(unittest.TestCase):
    """Integration tests for extended functionality."""
    
    def setUp(self):
        """Create full system."""
        self.manager = ModuleManager()
        register_all(self.manager)
        self.doc_gen = ModuleDocGenerator(self.manager)
        self.updater = ModuleUpdater(self.manager)
    
    def test_full_workflow(self):
        """Test complete workflow with all systems."""
        # 1. Generate documentation
        docs = self.doc_gen.generate_all_markdown()
        self.assertIsInstance(docs, str)
        self.assertGreater(len(docs), 100)
        
        # 2. Check for updates
        updates = self.updater.check_updates()
        self.assertIsInstance(updates, list)
        
        # 3. Load a module
        success = self.manager.load('memory')
        
        if success:
            # 4. Check health
            health = self.manager.health_check('memory')
            self.assertIsNotNone(health)
            
            # 5. Check all health
            all_health = self.manager.health_check_all()
            self.assertIn('memory', all_health)
    
    def test_documentation_after_loading(self):
        """Test that documentation reflects loaded modules."""
        # Load a module
        self.manager.load('memory')
        
        # Generate docs
        markdown = self.doc_gen.generate_markdown('memory')
        
        # Should show loaded state
        self.assertIn('loaded', markdown.lower())
    
    def test_health_monitoring_with_documentation(self):
        """Test combining health monitoring with doc generation."""
        # Load modules
        self.manager.load('memory')
        
        # Start health monitor
        self.manager.start_health_monitor(interval_seconds=1)
        
        # Generate documentation
        docs = self.doc_gen.generate_all_markdown()
        self.assertIsInstance(docs, str)
        
        # Stop monitor
        self.manager.stop_health_monitor()


if __name__ == '__main__':
    unittest.main()
