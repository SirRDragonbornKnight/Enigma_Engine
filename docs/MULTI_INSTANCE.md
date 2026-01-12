# Multi-Instance Support

## Overview

Enigma supports running multiple instances simultaneously with:
- Lock files to prevent conflicts
- Instance communication
- Resource management
- Instance monitoring

## Basic Usage

### Run with Instance ID

```bash
# Automatic instance ID
python run.py --run

# Specify instance ID
python run.py --run --instance my_instance_1

# Force new instance
python run.py --run --new-instance
```

### In Python Code

```python
from ai_tester.core.instance_manager import InstanceManager

# Create instance
with InstanceManager(instance_id="bot1") as manager:
    # Acquire model lock
    if manager.acquire_model_lock("my_model"):
        # Use the model safely
        # ... your code ...
        
        # Release lock when done
        manager.release_model_lock("my_model")
```

## Model Locking

Prevent multiple instances from using the same model:

```python
manager = InstanceManager()

# Try to acquire lock
if manager.acquire_model_lock("enigma"):
    print("Got the lock!")
    # Use model...
    manager.release_model_lock("enigma")
else:
    print("Model is being used by another instance")

# Wait for lock (with timeout)
if manager.acquire_model_lock("enigma", timeout=10.0):
    # Got it within 10 seconds
    pass
```

## Instance Communication

Send messages between instances:

```python
# Get all running instances
instances = manager.list_running_instances()
for instance in instances:
    print(f"{instance['instance_id']}: {instance['status']}")

# Send message to another instance
manager.send_to_instance(
    instance_id="bot2",
    message="Hello from bot1!"
)

# Receive messages
messages = manager.receive_messages()
for msg in messages:
    print(f"From {msg['from']}: {msg['message']}")
```

## Lock Storage

Instance locks are stored in `~/.ai_tester/locks/`:
- `instance_{id}.lock` - Instance registration
- `model_{name}.lock` - Model locks
- `messages/` - Inter-instance messages

## Automatic Cleanup

Stale locks (from crashed processes) are automatically removed:

```python
from ai_tester.core.instance_manager import cleanup_stale_locks

cleanup_stale_locks()
```

## Use Cases

### 1. Multiple Chatbots

```python
# Terminal 1
python run.py --run --instance chatbot1

# Terminal 2
python run.py --run --instance chatbot2
```

### 2. Training + Inference

```python
# Terminal 1: Training
python run.py --train --instance trainer

# Terminal 2: Chat (uses different model)
python run.py --run --instance chat
```

### 3. Web + API

```bash
# Terminal 1: Web dashboard
python run.py --web --instance web

# Terminal 2: Mobile API
python -c "from ai_tester.mobile.api import run_mobile_api; run_mobile_api()"
```

## Best Practices

1. **Always use unique instance IDs** for simultaneous instances
2. **Release locks** when done with a model
3. **Handle lock timeouts** gracefully
4. **Clean up** on shutdown (automatic with context manager)
5. **Monitor instances** regularly

## Troubleshooting

### "Could not acquire lock"
- Another instance is using the model
- Use `list_running_instances()` to see what's running
- Increase timeout or wait for other instance to finish

### Stale Locks
- Run cleanup: `cleanup_stale_locks()`
- Restart if a process crashed

### Message Not Received
- Check that target instance is running
- Messages expire after ~1 hour

## Example: Coordinated Training

```python
from ai_tester.core.instance_manager import InstanceManager

# Instance 1: Data preparation
with InstanceManager("prep") as manager:
    if manager.acquire_model_lock("dataset"):
        prepare_data()
        manager.send_to_instance("trainer", "data_ready")
        manager.release_model_lock("dataset")

# Instance 2: Training
with InstanceManager("trainer") as manager:
    messages = manager.receive_messages()
    for msg in messages:
        if msg['message'] == 'data_ready':
            if manager.acquire_model_lock("my_model"):
                train_model()
                manager.release_model_lock("my_model")
```

## See Also

- [Instance Manager API](../ai_tester/core/instance_manager.py)
- [Web Dashboard](WEB_MOBILE.md)
