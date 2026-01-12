#!/usr/bin/env python3
"""
Tests for the AI Tester communications module.

Run with: pytest tests/test_comms.py -v
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPIServer:
    """Tests for the Flask API server."""
    
    def test_create_app(self):
        """Test app creation."""
        from ai_tester.comms.api_server import create_app
        app = create_app()
        assert app is not None


class TestRemoteClient:
    """Tests for the remote client."""
    
    def test_client_init(self):
        """Test client initialization."""
        from ai_tester.comms.remote_client import RemoteClient
        client = RemoteClient("http://localhost:5000")
        assert client is not None
        assert client.base == "http://localhost:5000"


class TestNetwork:
    """Tests for network utilities."""
    
    def test_ai_tester_node_init(self):
        """Test AITesterNode initialization."""
        from ai_tester.comms.network import AITesterNode
        node = AITesterNode(port=0)  # Port 0 = auto-assign
        assert node is not None


class TestDiscovery:
    """Tests for network discovery."""
    
    def test_discovery_init(self):
        """Test discovery module initialization."""
        from ai_tester.comms.discovery import DeviceDiscovery
        service = DeviceDiscovery(node_name="test_node", node_port=5000)
        assert service is not None
        assert service.node_name == "test_node"


class TestMobileAPI:
    """Tests for mobile API."""
    
    def test_mobile_api_init(self):
        """Test MobileAPI initialization."""
        try:
            from ai_tester.comms.mobile_api import MobileAPI
            api = MobileAPI(port=5001)  # Use non-default port
            assert api is not None
            assert api.port == 5001
        except ImportError:
            pytest.skip("Flask not available for MobileAPI")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
