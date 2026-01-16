# Five-Star Features Guide

This guide covers all the new advanced features added to ForgeAI to achieve a five-star rating.

## Table of Contents

1. [3D Model Generation](#3d-model-generation)
2. [Enhanced Power Modes](#enhanced-power-modes)
3. [Motion Tracking & User Mimicry](#motion-tracking--user-mimicry)
4. [GGUF Model Support](#gguf-model-support)
5. [LoRA Fine-tuning](#lora-fine-tuning)
6. [Advanced Text Enhancement](#advanced-text-enhancement)
7. [Resource Monitoring](#resource-monitoring)

---

## 3D Model Generation

Generate 3D models from text descriptions using state-of-the-art AI models.

### Local Generation (Shap-E/Point-E)

```python
from forge_ai.modules import ModuleManager

manager = ModuleManager()
manager.load('threed_gen_local', config={
    'model': 'shap-e',  # or 'point-e'
    'guidance_scale': 15.0,
    'num_inference_steps': 64
})

# Generate 3D model
module = manager.get_module('threed_gen_local')
result = module.generate(
    "a red sports car",
    format="glb"
)
```

### Cloud Generation (Replicate API)

```python
manager.load('threed_gen_api', config={
    'api_key': 'your_replicate_key',
    'service': 'replicate'
})

module = manager.get_module('threed_gen_api')
result = module.generate("a modern house")
```

### Requirements

**Local:**
- GPU with 4GB+ VRAM
- Install: `pip install shap-e trimesh`

**Cloud:**
- Replicate API key
- Install: `pip install replicate`

---

## Enhanced Power Modes

Control resource usage with five optimized power modes.

### Available Modes

| Mode | CPU Threads | GPU Memory | Priority | Batch Size | Use Case |
|------|-------------|------------|----------|------------|----------|
| **Minimal** | 1 | 20% | Low | 1 | Heavy multitasking |
| **Gaming** | 2 | 30% | Low | 2 | Gaming + AI background |
| **Balanced** | Auto | 50% | Normal | 4 | Normal usage |
| **Performance** | Auto | 70% | Normal | 8 | Fast AI responses |
| **Maximum** | All | 90% | Normal | 16 | Maximum performance |

### Usage

```python
from forge_ai.core.resources import apply_resource_mode, get_resource_info

# Switch to gaming mode
apply_resource_mode("gaming")

# Check current settings
info = get_resource_info()
print(f"Mode: {info['mode']}")
print(f"CPU Threads: {info['torch_threads']}")
print(f"Batch Size Limit: {info['batch_size_limit']}")
```

### GUI Control

1. Open Settings Tab
2. Select mode from dropdown:
   - 🎮 Minimal
   - 🕹️ Gaming (NEW!)
   - ⚖️ Balanced
   - 🚀 Performance
   - 💪 Maximum
3. Click Apply

---

## Motion Tracking & User Mimicry

Real-time motion tracking for gesture control and avatar mimicry.

### Setup

```python
from forge_ai.modules import ModuleManager

manager = ModuleManager()
manager.load('motion_tracking', config={
    'camera_id': 0,
    'tracking_mode': 'holistic',  # pose, hands, face, or holistic
    'model_complexity': 1  # 0=fast, 1=default, 2=accurate
})

tracker = manager.get_module('motion_tracking')
tracker.start()
```

### Get Pose Data

```python
# Get latest pose
pose = tracker.get_pose()
if pose:
    print(f"Landmarks: {len(pose.landmarks)}")
    print(f"Timestamp: {pose.timestamp}")

# Recognize gestures
gesture = tracker.get_gesture()
if gesture:
    print(f"Detected: {gesture}")
```

### Requirements

- Webcam or camera
- Install: `pip install mediapipe opencv-python`

### Tracking Modes

- **pose**: Full body skeleton (33 landmarks)
- **hands**: Hand tracking (21 landmarks per hand)
- **face**: Face mesh (468 landmarks)
- **holistic**: All of the above combined

---

## GGUF Model Support

Load and run quantized GGUF models (llama.cpp compatible).

### Loading GGUF Models

```python
from forge_ai.core.gguf_loader import GGUFModel

model = GGUFModel(
    model_path="models/llama-7b-q4.gguf",
    n_ctx=2048,          # Context window
    n_gpu_layers=32,     # GPU layers (0 = CPU only)
    n_threads=4          # CPU threads
)

model.load()
```

### Text Generation

```python
# Standard generation
response = model.generate(
    "Once upon a time",
    max_tokens=100,
    temperature=0.8,
    top_p=0.95
)

# Streaming generation
response = model.generate(
    "Tell me a story",
    max_tokens=200,
    stream=True
)
```

### Chat Format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

response = model.chat(messages, max_tokens=150)
```

### Requirements

- Install: `pip install llama-cpp-python`
- For GPU support (CUDA): `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python`

### GPU Layer Recommendations

```python
from forge_ai.core.gguf_loader import recommend_gpu_layers

# Recommend layers based on model size and VRAM
layers = recommend_gpu_layers(
    model_size_gb=3.5,  # 3.5 GB model file
    vram_gb=8.0         # 8 GB VRAM available
)
print(f"Recommended GPU layers: {layers}")
```

---

## LoRA Fine-tuning

Efficient fine-tuning with Low-Rank Adaptation (LoRA).

### Benefits

- **10-100x fewer parameters** to train
- **Faster training** and less memory
- **Multiple adapters** for different tasks
- **Easy merging** into base model

### Quick Start

```python
from forge_ai.core.lora_utils import LoRATrainer, LoRAConfig

# Configure LoRA
config = LoRAConfig(
    rank=8,              # LoRA rank (4-64)
    alpha=16.0,          # Scaling factor
    target_modules=['q_proj', 'v_proj', 'k_proj'],
    learning_rate=3e-4,
    epochs=10
)

# Initialize trainer
trainer = LoRATrainer(model, config)

# Train
trainer.train(train_dataset, epochs=10)

# Save adapter (small file, ~10MB)
trainer.save_adapter("my_lora.pth")

# Or merge into base model
trainer.merge_and_save("merged_model.pth")
```

### Preparing Data

```python
from forge_ai.core.lora_utils import prepare_lora_dataset

dataset = prepare_lora_dataset(
    data_path="data/training.txt",
    tokenizer=tokenizer,
    max_length=512
)
```

### Data Format

```
Q: What is Python?
A: Python is a high-level programming language.

Q: How do I install packages?
A: Use pip: pip install package-name

User: Tell me a joke
AI: Why do programmers prefer dark mode? Because light attracts bugs!
```

---

## Advanced Text Enhancement

Automatic typo correction and smart suggestions.

### Typo Correction

```python
from forge_ai.utils.text_enhancement import correct_typos

text = "Teh modle is leraning form teh data"
corrected = correct_typos(text)
# "The model is learning from the data"
```

### Command Suggestions

```python
from forge_ai.utils.text_enhancement import suggest_command

commands = ["train", "inference", "generate", "evaluate"]
suggestion = suggest_command("trian", commands)
# Returns: "train"
```

### Parameter Validation

```python
from forge_ai.utils.text_enhancement import validate_parameter

is_valid, error = validate_parameter(
    value="150",
    param_type="int",
    min_val=1,
    max_val=100
)
# is_valid = False
# error = "Value must be <= 100"
```

### "Did You Mean" Suggestions

```python
from forge_ai.utils.text_enhancement import (
    find_closest_match, 
    format_did_you_mean
)

matches = find_closest_match("smal", ["small", "medium", "large"])
message = format_did_you_mean("smal", ["small", "medium", "large"])
# "Did you mean 'small', 'medium', or 'large'?"
```

---

## Resource Monitoring

Real-time resource usage dashboard.

### GUI Widget

```python
from forge_ai.gui.resource_monitor import ResourceMonitor

# Create monitor widget
monitor = ResourceMonitor()

# Record generation events
monitor.record_generation(
    tokens=50,
    latency=2.5
)
```

### Displays

- **CPU Usage**: Real-time percentage
- **Memory**: Used/Total with percentage
- **GPU**: Availability and usage
- **VRAM**: GPU memory usage
- **Performance**: Tokens/sec, latency, generation count
- **Current Mode**: Active power mode and settings

### Requirements

- Install: `pip install psutil`

---

## Best Practices

### For Gaming

1. Use **Gaming** or **Minimal** mode
2. Enable low priority in settings
3. Limit GPU memory to 30% or less
4. Set max 2 CPU threads

### For Training

1. Use **Performance** or **Maximum** mode
2. Use LoRA for faster fine-tuning
3. Monitor VRAM usage
4. Adjust batch size based on mode

### For Production

1. Use **Balanced** mode for reliability
2. Enable GGUF models for efficiency
3. Monitor resource usage
4. Use appropriate power mode for workload

---

## Troubleshooting

### 3D Generation Issues

**Problem**: Out of memory  
**Solution**: Use cloud API or upgrade GPU to 8GB+ VRAM

**Problem**: Slow generation  
**Solution**: Reduce num_inference_steps or use Point-E instead of Shap-E

### Motion Tracking Issues

**Problem**: Camera not detected  
**Solution**: Check camera_id, try 0, 1, 2, etc.

**Problem**: Low FPS  
**Solution**: Reduce model_complexity to 0

### GGUF Loading Issues

**Problem**: Model not loading  
**Solution**: Verify .gguf file is valid, check file permissions

**Problem**: Slow inference  
**Solution**: Increase n_gpu_layers if GPU available

### LoRA Training Issues

**Problem**: Out of memory  
**Solution**: Reduce rank, reduce batch_size

**Problem**: Poor results  
**Solution**: Increase rank, train longer, check data quality

---

## API Reference

See individual module documentation:

- `forge_ai.modules.registry` - Module definitions
- `forge_ai.core.gguf_loader` - GGUF loading
- `forge_ai.core.lora_utils` - LoRA training
- `forge_ai.core.resources` - Resource management
- `forge_ai.tools.motion_tracking` - Motion tracking
- `forge_ai.utils.text_enhancement` - Text utilities
- `forge_ai.gui.resource_monitor` - Resource monitoring widget

---

## Examples

Complete examples available in `examples/` directory:

- `examples/3d_generation_example.py`
- `examples/motion_tracking_example.py`
- `examples/gguf_loading_example.py`
- `examples/lora_training_example.py`

---

## Contributing

To add more features, follow the module system pattern:

1. Create module class in `forge_ai/modules/registry.py`
2. Add to `MODULE_REGISTRY` dict
3. Implement load/unload methods
4. Add tests
5. Update documentation

---

**ForgeAI** - The complete local AI framework
