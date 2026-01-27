# Deep Multi-Model Integration - Unified Orchestration System

## Overview

The Unified Orchestration System enables ForgeAI to coordinate multiple AI models and capabilities seamlessly. This allows:
- Using multiple specialized models together (chat + vision + code)
- Running everything on a single PC with models cooperating
- Models handing off tasks to each other automatically
- Using individual tools/modules WITH or WITHOUT an LLM
- Asynchronous task execution with background workers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MODEL ORCHESTRATOR                        │
│              (Central Intelligence Coordinator)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ CAPABILITY       │  │ MODEL POOL       │               │
│  │ REGISTRY         │  │ (Lazy Loading &  │               │
│  │ (What can each   │  │  LRU Eviction)   │               │
│  │  model do?)      │  │                  │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ COLLABORATION    │  │ TASK OFFLOADER   │               │
│  │ (Model-to-Model  │  │ (Async Tasks &   │               │
│  │  Communication)  │  │  Parallel Exec)  │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                             │
│  ┌──────────────────┐                                      │
│  │ STANDALONE       │                                      │
│  │ TOOLS            │                                      │
│  │ ("Without LLM")  │                                      │
│  └──────────────────┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Capability Registry (`capability_registry.py`)

Tracks what each model/tool can do.

**Capabilities:**
- `text_generation` - Generate text from prompts
- `code_generation` - Write and explain code
- `vision` - Understand images
- `image_generation` - Create images from text
- `audio_generation` - Generate audio/speech
- `speech_to_text` - Transcribe audio to text
- `text_to_speech` - Convert text to audio
- `embedding` - Create vector embeddings
- `reasoning` - Complex problem solving
- `tool_calling` - Execute function calls

**Example:**
```python
from forge_ai.core import get_capability_registry

registry = get_capability_registry()

# Register a model's capabilities
registry.register_model(
    model_id="forge:small",
    capabilities=["text_generation", "reasoning"],
    metadata={"size": "27M", "device": "cpu"}
)

# Find models with specific capability
models = registry.find_models_with_capability("code_generation")

# Find best model for a capability
best = registry.find_best_model("vision")
```

### 2. Model Pool (`model_pool.py`)

Manages loaded models efficiently with:
- **Lazy loading**: Load on first use
- **LRU eviction**: Unload least-used when memory tight
- **Resource tracking**: Monitor GPU/CPU memory usage
- **Preloading hints**: Load models before needed
- **Shared resources**: Share tokenizers and embeddings

**Example:**
```python
from forge_ai.core import get_model_pool

pool = get_model_pool()

# Load a model (lazy - happens on first use)
model = pool.get_model("forge:small")

# Preload models you'll need soon
pool.preload("huggingface:Qwen/Qwen2-1.5B-Instruct")

# Release a model (returns to pool, doesn't unload)
pool.release_model("forge:small")

# Get memory usage
usage = pool.get_memory_usage()
print(f"Total: {usage['total_mb']}MB, GPU: {usage['gpu_mb']}MB")

# Evict least-recently-used if memory is tight
pool.evict_lru(target_memory_mb=4000)
```

### 3. Model Collaboration (`collaboration.py`)

Enables models to work together and hand off tasks.

**Collaboration Types:**
- **Request/Response**: Simple ask-and-answer
- **Confidence Handoff**: Hand off if confidence too low
- **Pipeline**: Multi-stage processing through multiple models
- **Consensus**: Multiple models vote on answer

**Example:**
```python
from forge_ai.core import get_orchestrator

orchestrator = get_orchestrator()

# Model A asks Model B for help
response = orchestrator.collaborate(
    requesting_model="forge:small",
    target_capability="code_generation",
    task="Write a Python function to sort a list",
)

# Confidence-based handoff
# (if model confidence < 0.7, hand off to better model)
response = orchestrator.collaboration.smart_handoff(
    model_id="forge:tiny",
    capability="reasoning",
    task="Explain quantum computing",
    confidence_threshold=0.7,
)
```

### 4. Model Orchestrator (`orchestrator.py`)

Central coordinator for all AI models and capabilities.

**Features:**
- Register any model (Forge, HuggingFace, GGUF, external API)
- Route tasks based on capability matching
- Enable model-to-model communication
- Support fallback chains (if model A fails, try model B)
- Memory-aware loading (don't load what won't fit)
- Hot-swap models without restart

**Example:**
```python
from forge_ai.core import get_orchestrator

orchestrator = get_orchestrator()

# Register models
orchestrator.register_model(
    model_id="forge:small",
    capabilities=["text_generation", "reasoning"],
    load_args={"device": "cpu"}
)

orchestrator.register_model(
    model_id="huggingface:Qwen/Qwen2-VL-2B-Instruct",
    capabilities=["vision", "text_generation"],
    load_args={"device": "cuda"}
)

# Execute a task synchronously
result = orchestrator.execute_task(
    capability="vision",
    task="What's in this image?",
    parameters={"image_path": "photo.jpg"}
)

# Execute a task asynchronously (returns task ID)
task_id = orchestrator.execute_task(
    capability="code_generation",
    task="Write a Python function to sort a list",
    async_execution=True,
    priority=5,
    callback=lambda result: print(f"Code: {result}")
)

# Check async task status
status = orchestrator.get_async_task_status(task_id)
print(f"Task status: {status}")

# Wait for async task to complete
result = orchestrator.wait_for_async_task(task_id, timeout=30)

# Or use the convenience method
task_id = orchestrator.submit_async_task(
    capability="text_generation",
    task="Tell me a story",
    priority=3
)

# Set fallback chain
orchestrator.set_fallback_chain(
    "forge:small",
    ["forge:medium", "huggingface:Qwen/Qwen2-1.5B-Instruct"]
)

# Hot-swap a model
orchestrator.hot_swap_model(
    old_model_id="forge:small",
    new_model_id="forge:medium"
)

# Get status
status = orchestrator.get_status()
print(f"Loaded models: {status['loaded_models']}")
print(f"Memory usage: {status['memory_usage']}")
print(f"Async tasks: {status.get('task_offloader', {})}")
```

### 5. Task Offloader (`task_offloader.py`)

Enables asynchronous and parallel task execution with background workers.

**Features:**
- Priority-based task queue
- Background worker threads
- Progress tracking and callbacks
- Task cancellation
- Results caching

**Example:**
```python
from forge_ai.core import get_offloader, get_orchestrator

offloader = get_offloader()
orchestrator = get_orchestrator()
offloader.set_orchestrator(orchestrator)

# Submit async task
task_id = offloader.submit_task(
    capability="code_generation",
    task="Write a sorting function",
    priority=5,
    callback=lambda result: print(f"Done: {result}"),
    error_callback=lambda error: print(f"Error: {error}")
)

# Check task status
from forge_ai.core import TaskStatus
status = offloader.get_task_status(task_id)
if status == TaskStatus.COMPLETED:
    task = offloader.get_task(task_id)
    print(f"Result: {task.result}")

# Wait for completion
result = offloader.wait_for_task(task_id, timeout=30)

# Cancel a task
offloader.cancel_task(task_id)

# Get offloader status
status = offloader.get_status()
print(f"Workers: {status['num_workers']}")
print(f"Queue size: {status['queue_size']}")
print(f"Pending: {status['tasks']['pending']}")
print(f"Running: {status['tasks']['running']}")
print(f"Completed: {status['tasks']['completed']}")

# Clear pending tasks
cleared = offloader.clear_queue()

# Shutdown cleanly
offloader.shutdown(wait=True, timeout=10)
```

### 6. Standalone Tools (`standalone_tools.py`)

Use ForgeAI capabilities WITHOUT needing a full chat/LLM system.

**Available Tools:**
- `image` - Generate images
- `vision` - Analyze images
- `code` - Generate code
- `video` - Generate videos
- `audio` - Generate audio/music
- `tts` - Text to speech
- `stt` - Speech to text
- `embed` - Create text embeddings
- `3d` - Generate 3D models
- `avatar` - Control avatar
- `web` - Web search and browsing
- `file` - File operations

**Example:**
```python
from forge_ai import use_tool

# Generate an image without chat
image = use_tool("image", prompt="A sunset over mountains", width=512, height=512)

# Analyze an image without chat
description = use_tool("vision", image_path="photo.jpg", question="What's in this?")

# Generate code without chat
code = use_tool("code", prompt="Python function to sort a list", language="python")

# Text-to-speech without chat
use_tool("tts", text="Hello world", output_file="hello.wav")

# Speech-to-text without chat
text = use_tool("stt", audio_file="recording.wav")

# List all available tools
from forge_ai.core.standalone_tools import list_available_tools
tools = list_available_tools()
print(f"Available tools: {tools}")
```

## Configuration

Add to `forge_config.json` or use environment variables:

```json
{
  "orchestrator": {
    "default_chat_model": "auto",
    "default_code_model": "auto",
    "default_vision_model": "auto",
    "max_loaded_models": 3,
    "gpu_memory_limit_mb": 8000,
    "enable_collaboration": true,
    "enable_auto_fallback": true,
    "fallback_to_cpu": true,
    "enable_hot_swap": true,
    "enable_task_offloading": true
  },
  "task_offloader": {
    "num_workers": 2,
    "max_queue_size": 100,
    "enable_prioritization": true,
    "keep_history": true,
    "max_history_size": 1000,
    "auto_cleanup_seconds": 300,
    "enable_caching": false,
    "cache_ttl": 3600
  }
}
```

## Integration with Module Manager

The orchestration system automatically integrates with ForgeAI's module manager. When modules are loaded, they're automatically registered with the capability registry:

```python
from forge_ai.modules import ModuleManager

manager = ModuleManager()

# Load image generation module
manager.load('image_gen_local')  # Uses Stable Diffusion

# Module is automatically registered with orchestrator
# Now orchestrator knows this module provides "image_generation" capability
```

## Use Cases

### 1. Multi-Model Chat + Vision

```python
orchestrator = get_orchestrator()

# Register a chat model
orchestrator.register_model("forge:small", ["text_generation"])

# Register a vision model
orchestrator.register_model(
    "huggingface:llava-hf/llava-1.5-7b-hf",
    ["vision", "text_generation"]
)

# User sends image + question
# Orchestrator automatically routes image analysis to vision model
# Then uses chat model to format the response
```

### 2. Confidence-Based Handoff

```python
# Small model tries first
# If confidence < 70%, hands off to larger model
result = orchestrator.collaboration.smart_handoff(
    model_id="forge:tiny",
    capability="reasoning",
    task="Explain the theory of relativity",
    confidence_threshold=0.7,
)
```

### 3. Pipeline Processing

```python
# Multi-stage pipeline: translate → summarize → generate image
orchestrator.collaboration.execute_pipeline(
    stages=[
        {"capability": "translation", "model_id": "translator:en-fr"},
        {"capability": "summarization", "model_id": "forge:small"},
        {"capability": "image_generation", "model_id": "local:stable-diffusion"},
    ],
    initial_input="Long English text...",
)
```

### 4. Consensus Decision

```python
# Get 3 models to vote on the answer
result = orchestrator.collaboration.consensus(
    capability="reasoning",
    task="What is 2 + 2?",
    num_models=3,
    voting_strategy="majority",
)
```

### 5. Standalone Tool Usage

```python
# Use vision without loading a full chat model
description = use_tool(
    "vision",
    image_path="cat.jpg",
    question="What breed of cat is this?"
)

# Generate image without chat model
image = use_tool(
    "image",
    prompt="A futuristic city",
    width=768,
    height=768,
    provider="local"  # or "api"
)
```

## Memory Management

The orchestrator includes intelligent memory management:

1. **Lazy Loading**: Models only loaded when first needed
2. **LRU Eviction**: Least-recently-used models unloaded when memory is tight
3. **Resource Limits**: Configure GPU/CPU memory limits
4. **Auto-Eviction**: Automatically free memory when limits approached
5. **Fallback to CPU**: Move models to CPU if GPU memory is full

```python
# Configure memory limits
from forge_ai.core import OrchestratorConfig

config = OrchestratorConfig(
    max_loaded_models=3,
    gpu_memory_limit_mb=8000,
    cpu_memory_limit_mb=16000,
    fallback_to_cpu=True,
)

orchestrator = get_orchestrator(config)
```

## Performance Tips

1. **Preload frequently used models**: Use `pool.preload()` to load models before they're needed
2. **Set performance ratings**: Help orchestrator choose the best model for each task
3. **Configure fallback chains**: Provide backup options if primary model fails
4. **Use appropriate model sizes**: Don't load models that won't fit in memory
5. **Enable collaboration**: Let models help each other for better results
6. **Monitor memory usage**: Check `orchestrator.get_status()` regularly

## API Reference

See individual module documentation:
- `forge_ai/core/capability_registry.py`
- `forge_ai/core/model_pool.py`
- `forge_ai/core/collaboration.py`
- `forge_ai/core/orchestrator.py`
- `forge_ai/core/standalone_tools.py`

## Testing

Run orchestration tests:
```bash
python -m pytest tests/test_orchestration.py -v
```

## Future Enhancements

- [ ] Cross-device collaboration (Pi → PC)
- [ ] Distributed model loading across multiple GPUs
- [ ] Model performance profiling and auto-optimization
- [ ] Advanced scheduling algorithms
- [ ] Cost-aware routing for API models
- [ ] Quality-of-service guarantees
