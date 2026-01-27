"""
Test tunnel manager module.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from forge_ai.comms.tunnel_manager import (
    TunnelManager,
    TunnelProvider,
    TunnelConfig,
    get_tunnel_manager,
    create_tunnel
)


class TestTunnelManager:
    """Test TunnelManager class."""
    
    def test_init(self):
        """Test initialization."""
        manager = TunnelManager(provider="ngrok")
        assert manager.provider == TunnelProvider.NGROK
        assert not manager.is_running
        assert manager.tunnel_url is None
    
    def test_provider_validation(self):
        """Test provider validation."""
        # Valid providers
        for provider in ["ngrok", "localtunnel", "bore"]:
            manager = TunnelManager(provider=provider)
            assert manager.provider.value == provider
        
        # Invalid provider should raise ValueError
        with pytest.raises(ValueError):
            TunnelManager(provider="invalid")
    
    def test_get_tunnel_url_when_not_running(self):
        """Test getting URL when tunnel not running."""
        manager = TunnelManager()
        assert manager.get_tunnel_url() is None
    
    def test_is_tunnel_running(self):
        """Test tunnel running status."""
        manager = TunnelManager()
        assert not manager.is_tunnel_running()
    
    @patch('subprocess.run')
    @patch('subprocess.Popen')
    @patch('requests.get')
    def test_start_ngrok_tunnel(self, mock_get, mock_popen, mock_run):
        """Test starting ngrok tunnel."""
        # Mock ngrok version check
        mock_run.return_value = Mock(returncode=0)
        
        # Mock ngrok process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process
        
        # Mock ngrok API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tunnels": [
                {"proto": "https", "public_url": "https://abc123.ngrok.io"}
            ]
        }
        mock_get.return_value = mock_response
        
        manager = TunnelManager(provider="ngrok")
        url = manager.start_tunnel(port=5000)
        
        assert url == "https://abc123.ngrok.io"
        assert manager.is_running
        assert manager.tunnel_url == "https://abc123.ngrok.io"
    
    @patch('subprocess.run')
    def test_start_tunnel_ngrok_not_installed(self, mock_run):
        """Test error when ngrok not installed."""
        mock_run.side_effect = FileNotFoundError()
        
        manager = TunnelManager(provider="ngrok")
        url = manager.start_tunnel(port=5000)
        
        assert url is None
        assert not manager.is_running
    
    def test_stop_tunnel(self):
        """Test stopping tunnel."""
        manager = TunnelManager()
        manager.is_running = True
        manager.tunnel_url = "https://test.com"
        manager.tunnel_process = MagicMock()
        
        manager.stop_tunnel()
        
        assert not manager.is_running
        assert manager.tunnel_url is None
        manager.tunnel_process.terminate.assert_called_once()
    
    def test_singleton_get_tunnel_manager(self):
        """Test singleton pattern for get_tunnel_manager."""
        # Clear singleton
        import forge_ai.comms.tunnel_manager
        forge_ai.comms.tunnel_manager._tunnel_manager = None
        
        manager1 = get_tunnel_manager()
        manager2 = get_tunnel_manager()
        
        assert manager1 is manager2  # Same instance


class TestTunnelConfig:
    """Test TunnelConfig dataclass."""
    
    def test_tunnel_config_creation(self):
        """Test creating tunnel config."""
        config = TunnelConfig(
            provider=TunnelProvider.NGROK,
            port=5000,
            auth_token="test_token",
            region="us"
        )
        
        assert config.provider == TunnelProvider.NGROK
        assert config.port == 5000
        assert config.auth_token == "test_token"
        assert config.region == "us"
        assert config.auto_reconnect is True
    
    def test_tunnel_config_defaults(self):
        """Test default values."""
        config = TunnelConfig(
            provider=TunnelProvider.NGROK,
            port=5000
        )
        
        assert config.auth_token is None
        assert config.region is None
        assert config.subdomain is None
        assert config.auto_reconnect is True
        assert config.max_reconnect_attempts == 5


class TestHelperFunctions:
    """Test helper functions."""
    
    @patch('forge_ai.comms.tunnel_manager.TunnelManager.start_tunnel')
    def test_create_tunnel(self, mock_start):
        """Test create_tunnel helper function."""
        # Clear singleton
        import forge_ai.comms.tunnel_manager
        forge_ai.comms.tunnel_manager._tunnel_manager = None
        
        mock_start.return_value = "https://test.com"
        
        url = create_tunnel(port=5000, provider="ngrok")
        
        assert url == "https://test.com"
        mock_start.assert_called_once_with(5000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
