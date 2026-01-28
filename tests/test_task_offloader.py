"""
Tests for the Task Offloader system.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

from forge_ai.core.task_offloader import (
    TaskOffloader,
    OffloaderConfig,
    TaskStatus,
    OffloadedTask,
)


class TestTaskOffloader:
    """Tests for TaskOffloader class."""
    
    def test_initialization(self):
        """Test offloader initialization."""
        config = OffloaderConfig(
            num_workers=2,
            max_queue_size=10,
        )
        offloader = TaskOffloader(config)
        
        assert offloader.config.num_workers == 2
        assert offloader.config.max_queue_size == 10
        assert len(offloader._workers) == 2
    
    def test_submit_task(self):
        """Test submitting a task."""
        offloader = TaskOffloader()
        
        # Mock orchestrator
        mock_orch = Mock()
        mock_orch.execute_task = Mock(return_value="test result")
        offloader.set_orchestrator(mock_orch)
        
        # Submit task
        task_id = offloader.submit_task(
            capability="test",
            task="test task",
            priority=5,
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
        
        # Check task was stored
        task = offloader.get_task(task_id)
        assert task is not None
        assert task.capability == "test"
        assert task.task == "test task"
    
    def test_task_execution(self):
        """Test task execution by workers."""
        offloader = TaskOffloader()
        
        # Mock orchestrator
        mock_orch = Mock()
        mock_orch.execute_task = Mock(return_value="result")
        offloader.set_orchestrator(mock_orch)
        
        # Submit task
        task_id = offloader.submit_task(
            capability="test",
            task="test task",
        )
        
        # Wait for completion
        result = offloader.wait_for_task(task_id, timeout=5.0)
        
        assert result == "result"
        assert offloader.get_task_status(task_id) == TaskStatus.COMPLETED
        assert mock_orch.execute_task.called
    
    def test_task_priority(self):
        """Test task prioritization."""
        config = OffloaderConfig(
            num_workers=1,  # Single worker to test ordering
        )
        offloader = TaskOffloader(config)
        
        # Mock orchestrator that takes some time
        mock_orch = Mock()
        execution_order = []
        
        def slow_execute(capability, **kwargs):
            execution_order.append(kwargs.get("task"))
            time.sleep(0.1)
            return f"result for {kwargs.get('task')}"
        
        mock_orch.execute_task = slow_execute
        offloader.set_orchestrator(mock_orch)
        
        # Submit tasks with different priorities
        task_low = offloader.submit_task(
            capability="test",
            task="low priority",
            priority=10,
        )
        task_high = offloader.submit_task(
            capability="test",
            task="high priority",
            priority=1,
        )
        task_med = offloader.submit_task(
            capability="test",
            task="medium priority",
            priority=5,
        )
        
        # Wait for all to complete
        time.sleep(1.0)
        
        # High priority should execute first
        assert execution_order[0] == "high priority"
    
    def test_task_cancellation(self):
        """Test cancelling a task."""
        config = OffloaderConfig(
            num_workers=1,
        )
        offloader = TaskOffloader(config)
        
        # Mock orchestrator that takes time
        mock_orch = Mock()
        mock_orch.execute_task = Mock(
            side_effect=lambda **kwargs: time.sleep(0.5)
        )
        offloader.set_orchestrator(mock_orch)
        
        # Submit first task to occupy worker
        task_id1 = offloader.submit_task(
            capability="test",
            task="task 1",
        )
        
        # Submit second task that will be in queue
        task_id2 = offloader.submit_task(
            capability="test",
            task="task 2",
        )
        
        # Cancel the queued task
        time.sleep(0.1)  # Let first task start
        success = offloader.cancel_task(task_id2)
        
        assert success
        assert offloader.get_task_status(task_id2) == TaskStatus.CANCELLED
    
    def test_callback_on_success(self):
        """Test success callback is called."""
        offloader = TaskOffloader()
        
        # Mock orchestrator
        mock_orch = Mock()
        mock_orch.execute_task = Mock(return_value="test result")
        offloader.set_orchestrator(mock_orch)
        
        # Mock callback
        callback_called = []
        
        def callback(result):
            callback_called.append(result)
        
        # Submit task with callback
        task_id = offloader.submit_task(
            capability="test",
            task="test task",
            callback=callback,
        )
        
        # Wait for completion
        offloader.wait_for_task(task_id, timeout=5.0)
        
        # Check callback was called
        assert len(callback_called) == 1
        assert callback_called[0] == "test result"
    
    def test_error_callback(self):
        """Test error callback is called on failure."""
        offloader = TaskOffloader()
        
        # Mock orchestrator that raises error
        mock_orch = Mock()
        mock_orch.execute_task = Mock(
            side_effect=Exception("test error")
        )
        offloader.set_orchestrator(mock_orch)
        
        # Mock error callback
        errors = []
        
        def error_callback(error):
            errors.append(error)
        
        # Submit task with error callback
        task_id = offloader.submit_task(
            capability="test",
            task="test task",
            error_callback=error_callback,
        )
        
        # Wait and check status
        time.sleep(1.0)
        
        assert offloader.get_task_status(task_id) == TaskStatus.FAILED
        assert len(errors) == 1
        assert str(errors[0]) == "test error"
    
    def test_queue_management(self):
        """Test queue size and management."""
        config = OffloaderConfig(
            num_workers=1,
            max_queue_size=5,
        )
        offloader = TaskOffloader(config)
        
        # Mock orchestrator
        mock_orch = Mock()
        mock_orch.execute_task = Mock(return_value="result")
        offloader.set_orchestrator(mock_orch)
        
        # Submit multiple tasks
        task_ids = []
        for i in range(3):
            task_id = offloader.submit_task(
                capability="test",
                task=f"task {i}",
            )
            task_ids.append(task_id)
        
        # Check queue size
        assert offloader.get_queue_size() <= 3
        
        # Wait for completion
        time.sleep(1.0)
        
        # Clear any remaining
        cleared = offloader.clear_queue()
        assert cleared >= 0
    
    def test_get_status(self):
        """Test getting offloader status."""
        offloader = TaskOffloader()
        
        # Mock orchestrator
        mock_orch = Mock()
        mock_orch.execute_task = Mock(return_value="result")
        offloader.set_orchestrator(mock_orch)
        
        # Submit task
        task_id = offloader.submit_task(
            capability="test",
            task="test task",
        )
        
        # Get status
        status = offloader.get_status()
        
        assert "num_workers" in status
        assert "worker_info" in status
        assert "queue_size" in status
        assert "tasks" in status
        assert "statistics" in status
        
        assert status["num_workers"] > 0
        assert isinstance(status["tasks"], dict)
        assert isinstance(status["statistics"], dict)
    
    def test_shutdown(self):
        """Test clean shutdown."""
        offloader = TaskOffloader()
        
        # Submit tasks
        mock_orch = Mock()
        mock_orch.execute_task = Mock(return_value="result")
        offloader.set_orchestrator(mock_orch)
        
        task_id = offloader.submit_task(
            capability="test",
            task="test task",
        )
        
        # Shutdown
        offloader.shutdown(wait=True, timeout=5.0)
        
        # Check workers stopped
        assert len([w for w in offloader._workers if w.is_alive()]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
