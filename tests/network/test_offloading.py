"""Tests for network task offloading module."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from forge_ai.network import (
    RemoteOffloader,
    OffloadDecision,
    LoadBalancer,
    ServerInfo,
    BalancingStrategy,
    NetworkTaskQueue,
    NetworkTask,
    TaskPriority,
    FailoverManager,
    InferenceGateway,
)


class TestTaskPriority:
    """Tests for TaskPriority enum."""
    
    def test_priority_order(self):
        """Test task priority ordering."""
        assert TaskPriority.CRITICAL.value < TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value < TaskPriority.NORMAL.value
        assert TaskPriority.NORMAL.value < TaskPriority.LOW.value
        assert TaskPriority.LOW.value < TaskPriority.IDLE.value


class TestNetworkTask:
    """Tests for NetworkTask class."""
    
    def test_create_task(self):
        """Test creating a network task."""
        task = NetworkTask(
            capability="text_generation",
            payload={"prompt": "Hello, how are you?"},
            priority=TaskPriority.NORMAL,
        )
        
        assert task.task_id is not None
        assert task.capability == "text_generation"
        assert task.priority == TaskPriority.NORMAL
        assert task.status == "pending"
    
    def test_task_comparison(self):
        """Test task priority comparison for queue."""
        high = NetworkTask(
            capability="inference",
            priority=TaskPriority.HIGH,
        )
        low = NetworkTask(
            capability="inference",
            priority=TaskPriority.LOW,
        )
        
        # Lower priority value = higher priority
        assert high < low


class TestRemoteOffloader:
    """Tests for RemoteOffloader class."""
    
    def test_init(self):
        """Test initialization."""
        offloader = RemoteOffloader()
        
        assert offloader.prefer_local is True  # Default
        assert offloader._local_capability is not None
    
    def test_init_prefer_remote(self):
        """Test initialization with prefer_remote."""
        offloader = RemoteOffloader(prefer_local=False)
        
        assert offloader.prefer_local is False


class TestServerInfo:
    """Tests for ServerInfo class."""
    
    def test_create_server_info(self):
        """Test creating server info."""
        server = ServerInfo(
            address="192.168.1.100",
            port=8080,
            weight=2.0,
        )
        
        assert server.address == "192.168.1.100"
        assert server.port == 8080
        assert server.weight == 2.0
        assert server.is_healthy is True
    
    def test_server_key(self):
        """Test server key generation."""
        server = ServerInfo(address="localhost", port=5000)
        
        assert server.key == "localhost:5000"
    
    def test_error_rate(self):
        """Test error rate calculation."""
        server = ServerInfo(address="test", port=8080)
        
        # No requests yet
        assert server.error_rate == 0.0
        
        # Add some requests
        server.total_requests = 10
        server.total_errors = 2
        
        assert server.error_rate == 0.2


class TestLoadBalancer:
    """Tests for LoadBalancer class."""
    
    def test_init(self):
        """Test initialization."""
        balancer = LoadBalancer()
        
        assert balancer.strategy == BalancingStrategy.ADAPTIVE
    
    def test_init_with_strategy(self):
        """Test initialization with strategy."""
        balancer = LoadBalancer(strategy=BalancingStrategy.ROUND_ROBIN)
        
        assert balancer.strategy == BalancingStrategy.ROUND_ROBIN
    
    def test_add_server(self):
        """Test adding a server."""
        balancer = LoadBalancer()
        
        server = balancer.add_server("localhost", 8080, weight=1.5)
        
        assert server.address == "localhost"
        assert server.port == 8080
        assert server.weight == 1.5
        assert "localhost:8080" in balancer._servers
    
    def test_remove_server(self):
        """Test removing a server."""
        balancer = LoadBalancer()
        
        balancer.add_server("server1", 8080)
        balancer.add_server("server2", 8080)
        
        assert len(balancer._servers) == 2
        
        balancer.remove_server("server1", 8080)
        
        assert len(balancer._servers) == 1
        assert "server1:8080" not in balancer._servers


class TestNetworkTaskQueue:
    """Tests for NetworkTaskQueue class."""
    
    def test_init(self):
        """Test initialization."""
        queue = NetworkTaskQueue()
        
        assert queue.num_workers == 2  # Default
        assert queue.max_queue_size == 100  # Default
    
    def test_init_with_settings(self):
        """Test initialization with custom settings."""
        queue = NetworkTaskQueue(num_workers=4, max_queue_size=50)
        
        assert queue.num_workers == 4
        assert queue.max_queue_size == 50


class TestFailoverManager:
    """Tests for FailoverManager class."""
    
    def test_init(self):
        """Test initialization."""
        fm = FailoverManager()
        
        assert fm is not None


class TestInferenceGateway:
    """Tests for InferenceGateway class."""
    
    def test_init(self):
        """Test initialization."""
        gateway = InferenceGateway()
        
        assert gateway._offloader is not None
        assert gateway._load_balancer is not None
        # Failover is enabled by default
        assert gateway._failover is not None
        assert gateway.mode is not None
