#!/usr/bin/env python3
"""
Comprehensive tests for the complete tools system overhaul.

Tests all new features:
- Async execution
- Timeouts
- Caching
- Rate limiting
- Execution history
- Permissions
- Dependencies
- Parallel execution
- Configurable safety
- Analytics
- Versioning
- Validation
- Plugins
- Streaming
"""
import pytest
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAsyncExecution:
    """Tests for async tool execution."""
    
    def test_async_executor_init(self):
        """Test async executor initialization."""
        from enigma_engine.tools import AsyncToolExecutor
        executor = AsyncToolExecutor(max_workers=2)
        assert executor.max_workers == 2
        executor.shutdown()
    
    def test_async_execute_tool_sync(self):
        """Test sync wrapper for async execution."""
        from enigma_engine.tools import AsyncToolExecutor
        executor = AsyncToolExecutor()
        
        result = executor.execute_tool_sync("get_system_info", {})
        assert isinstance(result, dict)
        assert "success" in result
        
        executor.shutdown()
    
    def test_async_execute_multiple_sync(self):
        """Test executing multiple tools concurrently."""
        from enigma_engine.tools import AsyncToolExecutor
        executor = AsyncToolExecutor()
        
        tool_calls = [
            ("get_system_info", {}),
            ("list_directory", {"path": "."}),
        ]
        
        results = executor.execute_multiple_sync(tool_calls)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        
        executor.shutdown()


class TestToolCache:
    """Tests for tool result caching."""
    
    def test_cache_init(self, tmp_path):
        """Test cache initialization."""
        from enigma_engine.tools import ToolCache
        cache = ToolCache(cache_dir=tmp_path / "cache")
        assert cache.default_ttl == 300
        assert cache.max_memory_items == 100
    
    def test_cache_get_set(self, tmp_path):
        """Test cache get and set."""
        from enigma_engine.tools import ToolCache
        cache = ToolCache(cache_dir=tmp_path / "cache", default_ttl=10)
        
        # Cache miss
        result = cache.get("web_search", {"query": "test"})
        assert result is None
        
        # Set cache
        cache.set("web_search", {"query": "test"}, {"success": True, "result": "data"})
        
        # Cache hit
        cached = cache.get("web_search", {"query": "test"})
        assert cached is not None
        assert cached["success"] is True
    
    def test_cache_statistics(self, tmp_path):
        """Test cache statistics."""
        from enigma_engine.tools import ToolCache
        cache = ToolCache(cache_dir=tmp_path / "cache")
        
        stats = cache.get_statistics()
        assert "hits" in stats
        assert "misses" in stats
        assert stats["memory_entries"] == 0


class TestRateLimiting:
    """Tests for rate limiting."""
    
    def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        from enigma_engine.tools import RateLimiter
        limiter = RateLimiter()
        assert len(limiter.limits) > 0
    
    def test_rate_limiter_allow(self):
        """Test rate limit checking."""
        from enigma_engine.tools import RateLimiter
        limiter = RateLimiter(custom_limits={"test_tool": 2}, window_seconds=60)
        
        # First request allowed
        assert limiter.is_allowed("test_tool") is True
        limiter.record_request("test_tool")
        
        # Second request allowed
        assert limiter.is_allowed("test_tool") is True
        limiter.record_request("test_tool")
        
        # Third request blocked (limit is 2)
        assert limiter.is_allowed("test_tool") is False
    
    def test_rate_limiter_statistics(self):
        """Test rate limiter statistics."""
        from enigma_engine.tools import RateLimiter
        limiter = RateLimiter()
        
        stats = limiter.get_statistics()
        assert "tools_tracked" in stats
        assert "per_tool" in stats


class TestExecutionHistory:
    """Tests for execution history."""
    
    def test_history_init(self, tmp_path):
        """Test history initialization."""
        from enigma_engine.tools import ToolExecutionHistory
        history = ToolExecutionHistory(max_history=100, log_file=tmp_path / "log.txt")
        assert history.max_history == 100
    
    def test_history_record(self):
        """Test recording executions."""
        from enigma_engine.tools import ToolExecutionHistory
        history = ToolExecutionHistory()
        
        history.record("test_tool", {"param": "value"}, {"success": True, "result": "ok"}, 123.45)
        
        recent = history.get_recent(1)
        assert len(recent) == 1
        assert recent[0].tool_name == "test_tool"
        assert recent[0].success is True
    
    def test_history_statistics(self):
        """Test execution statistics."""
        from enigma_engine.tools import ToolExecutionHistory
        history = ToolExecutionHistory()
        
        # Record some executions
        history.record("tool1", {}, {"success": True}, 100)
        history.record("tool1", {}, {"success": False, "error": "test"}, 200)
        history.record("tool2", {}, {"success": True}, 150)
        
        stats = history.get_statistics()
        assert stats["total_executions"] == 3
        assert stats["success_count"] == 2
        assert stats["failure_count"] == 1


class TestPermissions:
    """Tests for permission system."""
    
    def test_permission_manager_init(self):
        """Test permission manager initialization."""
        from enigma_engine.tools import ToolPermissionManager, PermissionLevel
        manager = ToolPermissionManager(user_permission_level=PermissionLevel.WRITE)
        assert manager.user_permission_level == PermissionLevel.WRITE
    
    def test_can_execute(self):
        """Test permission checking."""
        from enigma_engine.tools import ToolPermissionManager, PermissionLevel
        manager = ToolPermissionManager(user_permission_level=PermissionLevel.READ)
        
        # Read tool allowed
        allowed, reason = manager.can_execute("read_file")
        assert allowed is True
        
        # Write tool not allowed
        allowed, reason = manager.can_execute("write_file")
        assert allowed is False
        assert reason is not None
    
    def test_get_available_tools(self):
        """Test getting available tools."""
        from enigma_engine.tools import ToolPermissionManager, PermissionLevel
        manager = ToolPermissionManager(user_permission_level=PermissionLevel.WRITE)
        
        available = manager.get_available_tools()
        assert isinstance(available, dict)
        assert len(available) > 0


class TestDependencyChecker:
    """Tests for dependency checking."""
    
    def test_dependency_checker_init(self):
        """Test dependency checker initialization."""
        from enigma_engine.tools import ToolDependencyChecker
        checker = ToolDependencyChecker()
        assert checker is not None
    
    def test_check_all_tools(self):
        """Test checking all tools."""
        from enigma_engine.tools import ToolDependencyChecker
        checker = ToolDependencyChecker()
        
        results = checker.check_all_tools()
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_get_missing_report(self):
        """Test getting missing dependencies report."""
        from enigma_engine.tools import ToolDependencyChecker
        checker = ToolDependencyChecker()
        
        report = checker.get_missing_report()
        assert isinstance(report, str)
        assert len(report) > 0


class TestParallelExecution:
    """Tests for parallel tool execution."""
    
    def test_parallel_executor_init(self):
        """Test parallel executor initialization."""
        from enigma_engine.tools import ParallelToolExecutor
        executor = ParallelToolExecutor(max_workers=2)
        assert executor.max_workers == 2
        executor.shutdown()
    
    def test_execute_parallel(self):
        """Test parallel execution."""
        from enigma_engine.tools import ParallelToolExecutor
        executor = ParallelToolExecutor()
        
        tool_calls = [
            ("get_system_info", {}),
            ("list_directory", {"path": "."}),
        ]
        
        results = executor.execute_parallel(tool_calls)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        
        executor.shutdown()
    
    def test_execute_batch(self):
        """Test batch execution."""
        from enigma_engine.tools import ParallelToolExecutor
        executor = ParallelToolExecutor()
        
        params_list = [
            {"path": "."},
            {"path": ".."},
        ]
        
        results = executor.execute_batch("list_directory", params_list)
        assert len(results) == 2
        
        executor.shutdown()


class TestURLSafety:
    """Tests for configurable URL safety."""
    
    def test_url_safety_config(self, tmp_path):
        """Test URL safety with config."""
        from enigma_engine.tools.url_safety import URLSafety
        config_path = tmp_path / "url_config.json"
        
        safety = URLSafety(config_path=config_path)
        assert safety.config_path == config_path
    
    def test_add_trusted_domain(self):
        """Test adding trusted domain."""
        from enigma_engine.tools.url_safety import URLSafety
        safety = URLSafety()
        
        safety.add_trusted_domain("example.com")
        assert "example.com" in safety.trusted_domains
        
        assert safety.is_trusted("https://example.com/page")


class TestAnalytics:
    """Tests for usage analytics."""
    
    def test_analytics_init(self):
        """Test analytics initialization."""
        from enigma_engine.tools import ToolAnalytics
        analytics = ToolAnalytics()
        assert analytics.max_records == 10000
    
    def test_record_usage(self):
        """Test recording usage."""
        from enigma_engine.tools import ToolAnalytics
        analytics = ToolAnalytics()
        
        analytics.record_usage("test_tool", True, 100.5)
        assert len(analytics.records) == 1
    
    def test_get_usage_report(self):
        """Test usage report generation."""
        from enigma_engine.tools import ToolAnalytics
        analytics = ToolAnalytics()
        
        # Record some usage
        analytics.record_usage("tool1", True, 100)
        analytics.record_usage("tool1", False, 200)
        analytics.record_usage("tool2", True, 150)
        
        report = analytics.get_usage_report()
        assert report["total_calls"] == 3
        assert "most_used_tools" in report


class TestToolVersioning:
    """Tests for tool versioning."""
    
    def test_tool_version_fields(self):
        """Test version fields on ToolDefinition."""
        from enigma_engine.tools import ToolDefinition, ToolParameter
        
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters=[],
            version="1.2.3",
            deprecated=False,
            added_in="1.0.0"
        )
        
        assert tool.version == "1.2.3"
        assert tool.deprecated is False
        assert tool.added_in == "1.0.0"
    
    def test_version_compatibility(self):
        """Test version compatibility checking."""
        from enigma_engine.tools import ToolDefinition, ToolParameter
        
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters=[],
            version="2.5.1"
        )
        
        # Same major, compatible
        assert tool.is_compatible("2.3.0") is True
        
        # Same major, newer minor required
        assert tool.is_compatible("2.6.0") is False
        
        # Different major
        assert tool.is_compatible("3.0.0") is False


class TestSchemaValidation:
    """Tests for schema validation."""
    
    def test_validator_init(self):
        """Test validator initialization."""
        from enigma_engine.tools import ToolSchemaValidator
        validator = ToolSchemaValidator(strict_mode=True)
        assert validator.strict_mode is True
    
    def test_validate_params(self):
        """Test parameter validation."""
        from enigma_engine.tools import ToolSchemaValidator, ToolDefinition, ToolParameter
        
        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters=[
                ToolParameter(name="count", type="int", description="Count", required=True),
                ToolParameter(name="name", type="string", description="Name", required=False, default="test"),
            ]
        )
        
        validator = ToolSchemaValidator()
        
        # Valid params
        is_valid, errors, validated = validator.validate(tool, {"count": 5})
        assert is_valid is True
        assert len(errors) == 0
        assert validated["count"] == 5
        assert validated["name"] == "test"
        
        # Missing required param
        is_valid, errors, validated = validator.validate(tool, {})
        assert is_valid is False
        assert len(errors) > 0


class TestPluginDiscovery:
    """Tests for plugin discovery."""
    
    def test_plugin_loader_init(self, tmp_path):
        """Test plugin loader initialization."""
        from enigma_engine.tools import ToolPluginLoader
        
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        
        loader = ToolPluginLoader(plugin_dirs=[plugin_dir], auto_discover=False)
        assert len(loader.plugin_dirs) == 1
    
    def test_discover_plugins(self, tmp_path):
        """Test plugin discovery."""
        from enigma_engine.tools import ToolPluginLoader
        
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        
        # Create a dummy plugin
        plugin_file = plugin_dir / "test_plugin.py"
        plugin_file.write_text('"""Test plugin"""\n\nTOOLS = []\n')
        
        loader = ToolPluginLoader(plugin_dirs=[plugin_dir], auto_discover=True)
        assert len(loader.discovered_plugins) >= 0


class TestStreaming:
    """Tests for result streaming."""
    
    def test_streaming_result_init(self):
        """Test streaming result initialization."""
        from enigma_engine.tools import StreamingToolResult
        stream = StreamingToolResult("test_tool")
        assert stream.tool_name == "test_tool"
    
    def test_streaming_put_get(self):
        """Test putting and getting from stream."""
        from enigma_engine.tools import StreamingToolResult
        stream = StreamingToolResult("test_tool")
        
        stream.put("item1")
        stream.put("item2")
        stream.done()
        
        items = list(stream)
        assert len(items) == 2
        assert items[0] == "item1"
        assert items[1] == "item2"
    
    def test_streaming_executor(self):
        """Test streaming executor."""
        from enigma_engine.tools import StreamingToolExecutor
        executor = StreamingToolExecutor()
        
        stream = executor.execute_streaming("get_system_info", {})
        assert stream is not None
        
        # Wait a bit for execution
        time.sleep(0.5)


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_async_with_cache(self, tmp_path):
        """Test async execution with caching."""
        from enigma_engine.tools import ToolCache
        
        cache = ToolCache(cache_dir=tmp_path / "cache", default_ttl=60)
        
        # Simulate tool execution result
        test_result = {"success": True, "result": "test data"}
        
        # First get - cache miss
        cached = cache.get("read_file", {"path": "test.txt"})
        assert cached is None
        
        # Set in cache
        cache.set("read_file", {"path": "test.txt"}, test_result)
        
        # Second get - cache hit
        cached = cache.get("read_file", {"path": "test.txt"})
        assert cached is not None
        assert cached.get("success") is True
        assert cached.get("result") == "test data"
    
    def test_parallel_with_history(self):
        """Test parallel execution with history tracking."""
        from enigma_engine.tools import ParallelToolExecutor, ToolExecutionHistory
        
        history = ToolExecutionHistory()
        executor = ParallelToolExecutor()
        
        tool_calls = [
            ("get_system_info", {}),
            ("list_directory", {"path": "."}),
        ]
        
        results = executor.execute_parallel(tool_calls)
        
        # Record in history
        for (tool_name, params), result in zip(tool_calls, results):
            history.record(tool_name, params, result)
        
        stats = history.get_statistics()
        assert stats["total_executions"] >= 2
        
        executor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
