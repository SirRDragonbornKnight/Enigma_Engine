# Universal Model Features - Troubleshooting & Validation

This guide helps you validate that all universal model features are working correctly and provides solutions to common issues.

## Quick Validation

Run the comprehensive validation script:

```bash
python validate_universal_model.py
```

This script will:
- ✅ Check all dependencies
- ✅ Test backward compatibility
- ✅ Validate RoPE scaling
- ✅ Test multi-modal integration
- ✅ Verify speculative decoding
- ✅ Check universal loading methods
- ✅ Validate LoRA support
- ✅ Test configuration features
- ✅ Run the full test suite
- ✅ Provide troubleshooting guidance

## Manual Testing

### 1. Basic Functionality Test

```python
from forge_ai.core.model import create_model
import torch

# Create model
model = create_model('small')
print(f"✓ Model created: {model.num_parameters:,} parameters")

# Test forward pass
input_ids = torch.randint(0, model.vocab_size, (1, 10))
logits = model(input_ids)
print(f"✓ Forward pass: {logits.shape}")

# Test generation
output = model.generate(input_ids, max_new_tokens=5)
print(f"✓ Generation: {output.shape}")
```

### 2. RoPE Scaling Test

```python
from forge_ai.core.model import ForgeConfig, Forge
import torch

# Test dynamic NTK scaling
config = ForgeConfig(
    vocab_size=1241,
    dim=256,
    n_layers=4,
    n_heads=4,
    max_seq_len=4096,
    rope_scaling_type='dynamic',
    rope_scaling_factor=4.0
)
model = Forge(config=config)
print(f"✓ Extended context: {config.max_seq_len} tokens")

# Test with long sequence
input_ids = torch.randint(0, 1241, (1, 512))
logits = model(input_ids)
print(f"✓ Long sequence: {logits.shape}")
```

### 3. Multi-Modal Test

```python
from forge_ai.core.model import ForgeConfig, Forge
import torch

config = ForgeConfig(
    vocab_size=1241,
    dim=256,
    n_layers=4,
    n_heads=4,
    vision_hidden_size=768
)
model = Forge(config=config)
print(f"✓ Vision projection: {model.vision_projection is not None}")

# Test forward
vision_features = torch.randn(1, 49, 768)
text_ids = torch.randint(0, 1241, (1, 20))
logits = model.forward_multimodal(
    input_ids=text_ids,
    vision_features=vision_features
)
print(f"✓ Multi-modal forward: {logits.shape}")
```

### 4. Run Test Suite

```bash
pytest tests/test_model.py tests/test_universal_model.py -v
```

Expected output: `54 passed, 2 skipped`

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError: No module named 'torch'

**Solution:**
```bash
# For CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: ModuleNotFoundError: No module named 'forge_ai'

**Solution:**
```bash
# Make sure you're in the project root directory
cd /path/to/Forge_AI

# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Or install in development mode
pip install -e .
```

### Issue 3: HuggingFace Model Loading Fails

**Error:**
```
ImportError: HuggingFace model loading requires transformers library.
```

**Solution:**
```bash
pip install transformers

# Then use:
from forge_ai.core.model import Forge
model = Forge.from_huggingface("microsoft/phi-2")
```

### Issue 4: Safetensors Loading Fails

**Error:**
```
ImportError: Safetensors loading requires safetensors library.
```

**Solution:**
```bash
pip install safetensors

# Then use:
from forge_ai.core.model import Forge
model = Forge.from_safetensors("model.safetensors")
```

### Issue 5: GGUF Loading Fails

**Error:**
```
ImportError: GGUF model loading requires gguf library.
```

**Solution:**
```bash
pip install gguf

# Then use:
from forge_ai.core.model import Forge
model = Forge.from_gguf("model.gguf")
```

### Issue 6: LoRA Adapters Don't Load

**Error:**
```
ImportError: LoRA support requires lora_utils module
```

**Solution:**

The `lora_utils` module needs to be implemented in `forge_ai/core/lora_utils.py`. Here's a minimal implementation:

```python
# forge_ai/core/lora_utils.py
import torch

def load_lora_weights(path):
    """Load LoRA weights from file."""
    return torch.load(path, map_location='cpu')

def apply_lora(model, lora_weights, merge=False, adapter_name=None):
    """Apply LoRA weights to model."""
    # Basic implementation - inject LoRA weights
    pass

def merge_lora_weights(model, lora_weights):
    """Merge LoRA weights into base model."""
    # Basic implementation - merge weights
    pass
```

### Issue 7: Out of Memory Error

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**

1. **Use smaller model:**
```python
model = create_model('tiny')  # or 'small', 'nano'
```

2. **Enable quantization:**
```python
model = create_model('small')
model.quantize('dynamic')  # or 'int8', 'int4'
```

3. **Use CPU:**
```python
model = create_model('small')
model.cpu()
```

4. **Reduce batch size or sequence length:**
```python
# Use smaller batches
input_ids = torch.randint(0, vocab_size, (1, 256))  # Instead of (4, 1024)
```

### Issue 8: RoPE Scaling Invalid Type Error

**Error:**
```
ValueError: rope_scaling_type must be one of {'linear', 'dynamic', 'yarn'}
```

**Solution:**
```python
# Correct usage:
config = ForgeConfig(
    rope_scaling_type='dynamic',  # Must be 'linear', 'dynamic', or 'yarn'
    rope_scaling_factor=2.0
)
```

### Issue 9: Multi-Modal Forward Error

**Error:**
```
ValueError: Vision features provided but vision_hidden_size not set in config
```

**Solution:**
```python
# Set vision_hidden_size in config:
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    vision_hidden_size=768  # Must match vision encoder output
)
model = Forge(config=config)

# Now you can use forward_multimodal:
logits = model.forward_multimodal(
    input_ids=text_ids,
    vision_features=vision_features
)
```

### Issue 10: Tests Fail

**Solution:**

1. **Install test dependencies:**
```bash
pip install pytest torch
```

2. **Run tests with verbose output:**
```bash
pytest tests/test_model.py tests/test_universal_model.py -v --tb=short
```

3. **Check specific failing test:**
```bash
pytest tests/test_universal_model.py::TestRoPEScaling::test_rope_scaling_model_creation -v
```

4. **Common test issues:**
   - Missing dependencies: Install torch, pytest
   - Import errors: Check PYTHONPATH
   - Memory errors: Use smaller test models

## Feature-Specific Validation

### Validate RoPE Scaling

```python
from forge_ai.core.model import ForgeConfig, Forge

# Test all three scaling types
for scaling_type in ['linear', 'dynamic', 'yarn']:
    config = ForgeConfig(
        vocab_size=1241,
        dim=128,
        n_layers=2,
        n_heads=4,
        rope_scaling_type=scaling_type,
        rope_scaling_factor=2.0
    )
    model = Forge(config=config)
    print(f"✓ {scaling_type} scaling works")
```

### Validate Multi-Modal

```python
from forge_ai.core.model import ForgeConfig, Forge
import torch

# Vision + Text
config = ForgeConfig(
    vocab_size=1241,
    dim=256,
    n_layers=4,
    n_heads=4,
    vision_hidden_size=768,
    audio_hidden_size=512
)
model = Forge(config=config)

vision = torch.randn(1, 49, 768)
audio = torch.randn(1, 100, 512)
text = torch.randint(0, 1241, (1, 20))

# Test all combinations
logits = model.forward_multimodal(input_ids=text)
print("✓ Text only")

logits = model.forward_multimodal(input_ids=text, vision_features=vision)
print("✓ Vision + Text")

logits = model.forward_multimodal(input_ids=text, audio_features=audio)
print("✓ Audio + Text")

logits = model.forward_multimodal(
    input_ids=text,
    vision_features=vision,
    audio_features=audio
)
print("✓ Vision + Audio + Text")
```

### Validate Speculative Decoding

```python
from forge_ai.core.model import create_model
import torch

draft = create_model('nano')
model = create_model('small')

# Enable
model.enable_speculative_decoding(draft, num_speculative_tokens=4)
print("✓ Speculative decoding enabled")

# Test generation
input_ids = torch.randint(0, model.vocab_size, (1, 10))
output = model.generate_speculative(input_ids, max_new_tokens=5)
print(f"✓ Generated {output.shape[1] - input_ids.shape[1]} tokens")

# Disable
model.disable_speculative_decoding()
print("✓ Speculative decoding disabled")
```

### Validate Universal Loading

```python
from forge_ai.core.model import Forge

# Check all methods exist
methods = [
    'from_any',
    'from_huggingface',
    'from_safetensors',
    'from_gguf',
    'from_onnx'
]

for method in methods:
    assert hasattr(Forge, method), f"Missing method: {method}"
    print(f"✓ Forge.{method}() exists")
```

## Performance Validation

### Benchmark RoPE Scaling

```python
import time
import torch
from forge_ai.core.model import ForgeConfig, Forge

# Standard model
config = ForgeConfig(vocab_size=1241, dim=256, n_layers=4, n_heads=4, max_seq_len=1024)
model = Forge(config=config)

# Extended context with RoPE scaling
config_extended = ForgeConfig(
    vocab_size=1241, dim=256, n_layers=4, n_heads=4,
    max_seq_len=4096,
    rope_scaling_type='dynamic',
    rope_scaling_factor=4.0
)
model_extended = Forge(config=config_extended)

# Benchmark
input_ids = torch.randint(0, 1241, (1, 512))

start = time.time()
logits = model(input_ids)
time_standard = time.time() - start

start = time.time()
logits = model_extended(input_ids)
time_extended = time.time() - start

print(f"Standard (1024 context): {time_standard:.4f}s")
print(f"Extended (4096 context): {time_extended:.4f}s")
print(f"Overhead: {(time_extended/time_standard - 1)*100:.1f}%")
```

### Benchmark Speculative Decoding

```python
import time
import torch
from forge_ai.core.model import create_model

draft = create_model('nano')
model = create_model('small')
input_ids = torch.randint(0, model.vocab_size, (1, 10))

# Standard generation
start = time.time()
output = model.generate(input_ids, max_new_tokens=50)
time_standard = time.time() - start

# Speculative generation
model.enable_speculative_decoding(draft, num_speculative_tokens=4)
start = time.time()
output = model.generate_speculative(input_ids, max_new_tokens=50)
time_speculative = time.time() - start

speedup = time_standard / time_speculative
print(f"Standard generation: {time_standard:.4f}s")
print(f"Speculative generation: {time_speculative:.4f}s")
print(f"Speedup: {speedup:.2f}x")
```

## Getting Help

1. **Run validation script:**
   ```bash
   python validate_universal_model.py
   ```

2. **Check documentation:**
   - `UNIVERSAL_MODEL_GUIDE.md` - Comprehensive usage guide
   - `UNIVERSAL_MODEL_IMPLEMENTATION.md` - Technical details
   - Inline docstrings in `forge_ai/core/model.py`

3. **Run interactive demo:**
   ```bash
   python examples/universal_model_demo.py
   ```

4. **Check test coverage:**
   ```bash
   pytest tests/test_universal_model.py -v --tb=short
   ```

5. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Quick Reference

| Feature | Config Parameter | Method |
|---------|-----------------|--------|
| Extended Context | `rope_scaling_type`, `rope_scaling_factor` | Standard forward/generate |
| Multi-Modal | `vision_hidden_size`, `audio_hidden_size` | `forward_multimodal()` |
| Speculative Decoding | N/A | `enable_speculative_decoding()`, `generate_speculative()` |
| Universal Loading | N/A | `from_any()`, `from_huggingface()`, etc. |
| LoRA Adapters | N/A | `load_lora()`, `merge_lora()` |
| MoE | `use_moe`, `num_experts` | Standard forward/generate |
| KV-Cache | `sliding_window`, `use_paged_attn`, `kv_cache_dtype` | Standard forward/generate |

## Success Indicators

✅ **All working correctly when:**
- Validation script passes all checks
- All 54 tests pass
- Basic model creation and generation works
- RoPE scaling accepts all three types
- Multi-modal forward passes with vision/audio
- Speculative decoding can be enabled/disabled
- All universal loading methods exist
- Config serialization/deserialization works

🔧 **Need attention if:**
- Import errors for core dependencies
- Tests fail unexpectedly
- Out of memory errors on small models
- Invalid configuration errors
- Missing method errors

📖 **More information:**
- See `UNIVERSAL_MODEL_GUIDE.md` for usage examples
- See inline docstrings for API documentation
- Run `python examples/universal_model_demo.py` for interactive demo
