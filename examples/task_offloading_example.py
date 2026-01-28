"""
Example: Task Offloading with ForgeAI Orchestrator

This example demonstrates how to use the task offloading system
for asynchronous and parallel task execution.
"""

from forge_ai.core import (
    get_orchestrator,
    get_offloader,
    OrchestratorConfig,
    OffloaderConfig,
    TaskStatus,
)


def example_basic_async():
    """Basic asynchronous task execution."""
    print("=" * 60)
    print("Example 1: Basic Async Execution")
    print("=" * 60)
    
    # Create orchestrator with task offloading enabled
    config = OrchestratorConfig(
        enable_task_offloading=True,
        offloader_config=OffloaderConfig(
            num_workers=2,
            max_queue_size=100,
        ),
    )
    orchestrator = get_orchestrator(config)
    
    # Register a simple model
    # (In real usage, you'd register actual models)
    orchestrator.register_model(
        model_id="test:model",
        capabilities=["text_generation"],
    )
    
    # Execute task asynchronously
    task_id = orchestrator.execute_task(
        capability="text_generation",
        task="Tell me a joke",
        async_execution=True,
        priority=5,
    )
    
    print(f"Task submitted: {task_id}")
    
    # Check status
    status = orchestrator.get_async_task_status(task_id)
    print(f"Task status: {status}")
    
    # Wait for result
    try:
        result = orchestrator.wait_for_async_task(task_id, timeout=10)
        print(f"Task result: {result}")
    except Exception as e:
        print(f"Task failed: {e}")


def example_parallel_tasks():
    """Execute multiple tasks in parallel."""
    print("\n" + "=" * 60)
    print("Example 2: Parallel Task Execution")
    print("=" * 60)
    
    orchestrator = get_orchestrator()
    
    # Submit multiple tasks
    task_ids = []
    for i in range(5):
        task_id = orchestrator.submit_async_task(
            capability="text_generation",
            task=f"Generate text #{i}",
            priority=i,  # Different priorities
        )
        task_ids.append(task_id)
        print(f"Submitted task {i}: {task_id}")
    
    # Wait for all to complete
    print("\nWaiting for tasks to complete...")
    results = []
    for i, task_id in enumerate(task_ids):
        try:
            result = orchestrator.wait_for_async_task(task_id, timeout=10)
            results.append((i, result))
            print(f"Task {i} completed: {result}")
        except Exception as e:
            print(f"Task {i} failed: {e}")
    
    print(f"\nCompleted {len(results)} tasks")


def example_with_callbacks():
    """Use callbacks for async task completion."""
    print("\n" + "=" * 60)
    print("Example 3: Callbacks")
    print("=" * 60)
    
    orchestrator = get_orchestrator()
    results_list = []
    errors_list = []
    
    def on_success(result):
        results_list.append(result)
        print(f"✓ Task completed: {result}")
    
    def on_error(error):
        errors_list.append(error)
        print(f"✗ Task failed: {error}")
    
    # Submit task with callbacks
    task_id = orchestrator.execute_task(
        capability="text_generation",
        task="Generate a story",
        async_execution=True,
        callback=on_success,
        error_callback=on_error,
    )
    
    print(f"Task submitted with callbacks: {task_id}")
    
    # Give it time to complete
    import time
    time.sleep(2)
    
    print(f"\nResults: {len(results_list)} successful, {len(errors_list)} failed")


def example_task_cancellation():
    """Cancel pending tasks."""
    print("\n" + "=" * 60)
    print("Example 4: Task Cancellation")
    print("=" * 60)
    
    orchestrator = get_orchestrator()
    
    # Submit tasks
    task_ids = []
    for i in range(3):
        task_id = orchestrator.submit_async_task(
            capability="text_generation",
            task=f"Long running task {i}",
        )
        task_ids.append(task_id)
        print(f"Submitted task {i}: {task_id}")
    
    # Cancel some tasks
    import time
    time.sleep(0.5)
    
    cancelled_count = 0
    for i, task_id in enumerate(task_ids[1:], 1):  # Skip first
        if orchestrator.cancel_async_task(task_id):
            cancelled_count += 1
            print(f"✓ Cancelled task {i}")
        else:
            print(f"✗ Could not cancel task {i}")
    
    print(f"\nCancelled {cancelled_count} tasks")


def example_offloader_status():
    """Check offloader status and statistics."""
    print("\n" + "=" * 60)
    print("Example 5: Offloader Status")
    print("=" * 60)
    
    orchestrator = get_orchestrator()
    
    # Submit some tasks
    for i in range(5):
        orchestrator.submit_async_task(
            capability="text_generation",
            task=f"Task {i}",
        )
    
    # Get orchestrator status
    status = orchestrator.get_status()
    
    print("Orchestrator Status:")
    print(f"  Registered models: {len(status.get('registered_models', []))}")
    print(f"  Loaded models: {len(status.get('loaded_models', []))}")
    
    # Get task offloader status
    if 'task_offloader' in status:
        offloader_status = status['task_offloader']
        print("\nTask Offloader Status:")
        print(f"  Workers: {offloader_status['num_workers']}")
        print(f"  Queue size: {offloader_status['queue_size']}")
        print(f"  Pending: {offloader_status['tasks']['pending']}")
        print(f"  Running: {offloader_status['tasks']['running']}")
        print(f"  Completed: {offloader_status['tasks']['completed']}")
        print(f"  Failed: {offloader_status['tasks']['failed']}")
        
        # Statistics
        stats = offloader_status['statistics']
        print("\nStatistics:")
        print(f"  Total submitted: {stats['total_submitted']}")
        print(f"  Total completed: {stats['total_completed']}")
        print(f"  Total failed: {stats['total_failed']}")


if __name__ == "__main__":
    # Note: These examples require actual models to be registered
    # In real usage, you would register Forge models or HuggingFace models
    
    print("\n🚀 ForgeAI Task Offloading Examples\n")
    print("Note: These examples require models to be registered.")
    print("They demonstrate the API but may not execute without models.\n")
    
    try:
        example_basic_async()
        example_parallel_tasks()
        example_with_callbacks()
        example_task_cancellation()
        example_offloader_status()
    except Exception as e:
        print(f"\n⚠️  Example failed (expected without models): {e}")
    
    print("\n✅ Examples complete!")
