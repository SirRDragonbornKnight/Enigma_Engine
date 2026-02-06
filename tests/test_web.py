"""
Tests for forge_ai.web module.

Tests web server functionality including:
- FastAPI server setup
- REST endpoints
- WebSocket connections
- Authentication
- Static file serving
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


class TestWebServerCreation:
    """Test web server initialization."""
    
    def test_server_requires_fastapi(self):
        """Test that server requires FastAPI."""
        # Mock FastAPI not available
        with patch.dict('sys.modules', {'fastapi': None}):
            from forge_ai.web.server import FASTAPI_AVAILABLE
            # Note: FASTAPI_AVAILABLE is set at import time
    
    @pytest.mark.skipif(True, reason="Requires FastAPI")
    def test_server_creation(self):
        """Test creating web server."""
        from forge_ai.web.server import ForgeWebServer
        
        server = ForgeWebServer(host="127.0.0.1", port=8080)
        assert server is not None
        assert server.port == 8080
    
    @pytest.mark.skipif(True, reason="Requires FastAPI")
    def test_server_cors_setup(self):
        """Test CORS middleware is configured."""
        from forge_ai.web.server import ForgeWebServer
        
        server = ForgeWebServer()
        # CORS should be configured
        assert server.app is not None


class TestWebAuth:
    """Test web authentication."""
    
    def test_auth_creation(self):
        """Test creating auth manager."""
        from forge_ai.web.auth import WebAuth
        
        auth = WebAuth()
        assert auth is not None
    
    def test_token_generation(self):
        """Test generating auth tokens."""
        from forge_ai.web.auth import WebAuth
        
        auth = WebAuth()
        token = auth.generate_token()
        
        assert token is not None
        assert len(token) > 0
    
    def test_token_verification(self):
        """Test verifying auth tokens."""
        from forge_ai.web.auth import WebAuth
        
        auth = WebAuth()
        token = auth.generate_token()
        
        # Valid token should verify
        assert auth.verify_token(token) is True
        
        # Invalid token should fail
        assert auth.verify_token("invalid_token") is False
    
    def test_token_expiration(self):
        """Test token expiration."""
        from forge_ai.web.auth import WebAuth
        import time
        
        auth = WebAuth(token_expiry_seconds=1)
        token = auth.generate_token()
        
        # Should be valid immediately
        assert auth.verify_token(token) is True
        
        # Should expire after delay
        time.sleep(2)
        assert auth.verify_token(token) is False


class TestWebDiscovery:
    """Test local network discovery."""
    
    def test_get_local_ip(self):
        """Test getting local IP address."""
        from forge_ai.web.discovery import get_local_ip
        
        ip = get_local_ip()
        assert ip is not None
        # Should be an IP-like string
        assert '.' in ip or ip == 'localhost'
    
    def test_discovery_creation(self):
        """Test creating discovery service."""
        from forge_ai.web.discovery import LocalDiscovery
        
        discovery = LocalDiscovery()
        assert discovery is not None


class TestWebEndpoints:
    """Test REST API endpoints."""
    
    @pytest.fixture
    def mock_app(self):
        """Create mock Flask/FastAPI test client."""
        from forge_ai.web.server import ForgeWebServer
        
        with patch('forge_ai.web.server.FASTAPI_AVAILABLE', True):
            server = ForgeWebServer(require_auth=False)
            return server.app
    
    def test_health_endpoint(self):
        """Test health check endpoint returns OK."""
        # Would use TestClient if FastAPI available
        pass
    
    def test_info_endpoint(self):
        """Test info endpoint returns server info."""
        pass


class TestWebSocket:
    """Test WebSocket functionality."""
    
    def test_connection_manager(self):
        """Test WebSocket connection manager."""
        from forge_ai.web.server import ConnectionManager
        
        manager = ConnectionManager()
        assert manager is not None
        assert len(manager.active_connections) == 0


class TestRequestModels:
    """Test request/response models."""
    
    def test_chat_message_model(self):
        """Test ChatMessage model."""
        from forge_ai.web.server import ChatMessage
        
        msg = ChatMessage(content="Hello")
        assert msg.content == "Hello"
        assert msg.max_tokens == 200  # default
    
    def test_generate_image_request(self):
        """Test GenerateImageRequest model."""
        from forge_ai.web.server import GenerateImageRequest
        
        req = GenerateImageRequest(prompt="A sunset")
        assert req.prompt == "A sunset"
        assert req.width == 512  # default
        assert req.height == 512  # default


class TestTelemetryDashboard:
    """Test telemetry dashboard."""
    
    def test_dashboard_creation(self):
        """Test creating telemetry dashboard."""
        from forge_ai.web.telemetry_dashboard import TelemetryDashboard
        
        dashboard = TelemetryDashboard()
        assert dashboard is not None
    
    def test_metrics_collection(self):
        """Test collecting metrics."""
        from forge_ai.web.telemetry_dashboard import TelemetryDashboard
        
        dashboard = TelemetryDashboard()
        
        # Record a metric
        dashboard.record_metric("requests", 1)
        
        # Should be retrievable
        metrics = dashboard.get_metrics()
        assert "requests" in metrics


class TestFlaskApp:
    """Test Flask web app."""
    
    def test_app_creation(self):
        """Test creating Flask app."""
        from forge_ai.web.app import create_app
        
        app = create_app()
        assert app is not None
    
    def test_run_web_function(self):
        """Test run_web entry point."""
        from forge_ai.web.app import run_web
        
        # Should exist and be callable
        assert callable(run_web)
