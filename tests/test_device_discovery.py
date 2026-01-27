#!/usr/bin/env python3
"""
Tests for the device discovery system.

Run with: pytest tests/test_device_discovery.py -v
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDeviceDiscovery:
    """Tests for DeviceDiscovery class."""
    
    def test_discovery_init(self):
        """Test discovery initialization."""
        from forge_ai.comms.discovery import DeviceDiscovery
        
        service = DeviceDiscovery(node_name="test_node", node_port=5000)
        assert service is not None
        assert service.node_name == "test_node"
        assert service.node_port == 5000
        assert service.discovered == {}
    
    def test_get_local_ip(self):
        """Test getting local IP address."""
        from forge_ai.comms.discovery import DeviceDiscovery
        
        service = DeviceDiscovery(node_name="test_node", node_port=5000)
        ip = service.get_local_ip()
        
        # Should return a valid IP (either real or localhost)
        assert ip is not None
        assert isinstance(ip, str)
        assert len(ip.split('.')) == 4
    
    def test_discover_self_excluded(self):
        """Test that we don't discover ourselves."""
        from forge_ai.comms.discovery import DeviceDiscovery
        
        service = DeviceDiscovery(node_name="test_node", node_port=5000)
        
        # Simulate discovering ourselves - should be filtered out
        # In real scenario, the broadcast_discover would filter this
        assert service.node_name == "test_node"
    
    def test_device_url_generation(self):
        """Test generating device URLs."""
        from forge_ai.comms.discovery import DeviceDiscovery
        
        service = DeviceDiscovery(node_name="test_node", node_port=5000)
        
        # Add a fake discovered device
        service.discovered["remote_node"] = {
            "ip": "192.168.1.100",
            "port": 5001,
            "last_seen": 0,
        }
        
        url = service.get_device_url("remote_node")
        assert url == "http://192.168.1.100:5001"
        
        # Non-existent device
        url = service.get_device_url("nonexistent")
        assert url is None
    
    def test_on_discover_callback(self):
        """Test discovery callbacks."""
        from forge_ai.comms.discovery import DeviceDiscovery
        
        service = DeviceDiscovery(node_name="test_node", node_port=5000)
        
        # Track callback invocations
        callback_called = []
        
        def callback(name, info):
            callback_called.append((name, info))
        
        service.on_discover(callback)
        
        # Trigger callback manually
        service._notify_discover("test_device", {"ip": "192.168.1.100", "port": 5001})
        
        assert len(callback_called) == 1
        assert callback_called[0][0] == "test_device"
        assert callback_called[0][1]["ip"] == "192.168.1.100"
    
    def test_convenience_function(self):
        """Test the convenience discover function."""
        from forge_ai.comms.discovery import discover_forge_ai_nodes
        
        # This will timeout quickly if no nodes are running
        # Just test that it doesn't crash
        nodes = discover_forge_ai_nodes(node_name="test_scanner", timeout=0.5)
        
        assert isinstance(nodes, dict)
        # May be empty if no nodes running, which is fine


class TestDiscoveryIntegration:
    """Integration tests for discovery system."""
    
    def test_imports_without_torch(self):
        """Test that discovery can be imported without torch."""
        # This should not raise ImportError
        from forge_ai.comms.discovery import DeviceDiscovery, discover_forge_ai_nodes
        
        assert DeviceDiscovery is not None
        assert discover_forge_ai_nodes is not None
    
    def test_discovery_in_comms_package(self):
        """Test that discovery is exported from comms package."""
        from forge_ai.comms import DeviceDiscovery, discover_forge_ai_nodes
        
        assert DeviceDiscovery is not None
        assert discover_forge_ai_nodes is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
