# Universal Model Features - Usage Examples

This document demonstrates the new universal features added to `enigma_engine/core/model.py`.

## Table of Contents
1. [Backward Compatibility](#backward-compatibility)
2. [RoPE Scaling for Extended Context](#rope-scaling)
3. [Universal Model Loading](#universal-loading)
4. [Multi-Modal Integration](#multi-modal)
5. [LoRA Adapters](#lora-adapters)
6. [Speculative Decoding](#speculative-decoding)
7. [Enhanced KV-Cache](#enhanced-kv-cache)
8. [Mixture of Experts](#mixture-of-experts)

---

## Backward Compatibility

All existing code continues to work unchanged:

```python
from enigma_engine.core.model import create_model, Forge, ForgeConfig

# Standard model creation (unchanged)
model = create_model('small')
output = model.generate(input_ids, max_new_tokens=50)

# Config-based creation (unchanged)
config = ForgeConfig(vocab_size=8000, dim=512, n_layers=8)
model = Forge(config=config)
```

---

## RoPE Scaling

Extend context length beyond training with three scaling methods:

### Linear Scaling
```python
from enigma_engine.core.model import ForgeConfig, Forge

# Extend context by 2x with linear scaling
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    max_seq_len=4096,  # Extended from 2048
    rope_scaling_type="linear",
    rope_scaling_factor=2.0
)
model = Forge(config=config)
```

### Dynamic NTK Scaling
Better quality for moderate extensions:

```python
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    max_seq_len=8192,  # 4x extension
    rope_scaling_type="dynamic",
    rope_scaling_factor=4.0
)
model = Forge(config=config)
```

### YaRN Scaling
Best for very long contexts:

```python
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    max_seq_len=16384,  # 8x extension
    rope_scaling_type="yarn",
    rope_scaling_factor=8.0
)
model = Forge(config=config)
```

---

## Universal Loading

Load models from any format:

### Auto-Detection
```python
from enigma_engine.core.model import Forge

# Automatically detects format
model = Forge.from_any("path/to/model")
model = Forge.from_any("model.gguf")
model = Forge.from_any("model.safetensors")
model = Forge.from_any("microsoft/phi-2")  # HuggingFace ID
```

### HuggingFace Models
```python
# Load from HuggingFace Hub
model = Forge.from_huggingface("gpt2")
model = Forge.from_huggingface("microsoft/phi-2")

# Load local HuggingFace format
model = Forge.from_huggingface("path/to/local/model")
```

**Requirements**: `pip install transformers`

### Safetensors Format
```python
# Safer and faster than pickle
model = Forge.from_safetensors("model.safetensors")
```

**Requirements**: `pip install safetensors`

### GGUF Format (llama.cpp)
```python
# Load quantized GGUF models
model = Forge.from_gguf("llama-2-7b.Q4_K_M.gguf")
```

**Requirements**: `pip install gguf`

---

## Multi-Modal Integration

Integrate vision and audio with text models:

### Vision + Text Model
```python
from enigma_engine.core.model import ForgeConfig, Forge
import torch

# Configure with vision encoder dimension
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    vision_hidden_size=768  # From vision encoder
)
model = Forge(config=config)

# Forward pass with vision + text
vision_features = torch.randn(1, 196, 768)  # From vision encoder
text_ids = torch.randint(0, 8000, (1, 50))

logits = model.forward_multimodal(
    input_ids=text_ids,
    vision_features=vision_features
)
```

### Audio + Text Model
```python
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    audio_hidden_size=512  # From audio encoder
)
model = Forge(config=config)

# Forward pass with audio + text
audio_features = torch.randn(1, 100, 512)
text_ids = torch.randint(0, 8000, (1, 50))

logits = model.forward_multimodal(
    input_ids=text_ids,
    audio_features=audio_features
)
```

### Vision + Audio + Text
```python
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    vision_hidden_size=768,
    audio_hidden_size=512
)
model = Forge(config=config)

# All three modalities
logits = model.forward_multimodal(
    input_ids=text_ids,
    vision_features=vision_features,
    audio_features=audio_features
)
```

---

## LoRA Adapters

Low-Rank Adaptation for efficient fine-tuning:

### Load LoRA Adapter
```python
from enigma_engine.core.model import create_model

model = create_model('medium')

# Load LoRA adapter
model.load_lora("path/to/lora_adapter.pth", adapter_name="coding")

# Use model with adapter
output = model.generate(input_ids)
```

### Multiple Adapters
```python
# Load multiple adapters for different tasks
model.load_lora("adapters/coding.pth", "coding")
model.load_lora("adapters/creative.pth", "creative")
model.load_lora("adapters/factual.pth", "factual")
```

### Merge Adapters
```python
# Merge specific adapter into base weights
model.merge_lora("coding")

# Or merge all adapters
model.merge_lora()
```

**Requirements**: Implemented via `enigma_engine/core/lora_utils.py`

---

## Speculative Decoding

2-4x faster generation with draft model:

### Enable Speculative Decoding
```python
from enigma_engine.core.model import create_model

# Create small draft model and large main model
draft_model = create_model('tiny')
main_model = create_model('large')

# Enable speculative decoding
main_model.enable_speculative_decoding(
    draft_model,
    num_speculative_tokens=4  # Draft 4 tokens at once
)

# Generate (automatically uses speculation)
output = main_model.generate_speculative(
    input_ids,
    max_new_tokens=100
)
```

### How It Works
1. Draft model generates 4 tokens quickly (small = fast)
2. Main model verifies all 4 tokens in one pass (parallel!)
3. Accept correct tokens, reject and regenerate incorrect ones
4. Result: 2-4x speedup with same quality

### Disable When Not Needed
```python
main_model.disable_speculative_decoding()
```

---

## Enhanced KV-Cache

Improved memory management and efficiency:

### Sliding Window Attention
```python
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    sliding_window=2048  # Only attend to last 2048 tokens
)
model = Forge(config=config)
```

### Paged Attention
Better memory management for long contexts:

```python
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    use_paged_attn=True  # Enable paged attention
)
model = Forge(config=config)
```

### Quantized KV-Cache
Save memory by quantizing cache:

```python
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    kv_cache_dtype="int8"  # Quantize cache to INT8
)
model = Forge(config=config)
```

---

## Mixture of Experts

Conditional computation with expert routing:

### Basic MoE Configuration
```python
from enigma_engine.core.model import ForgeConfig, Forge

config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    use_moe=True,
    num_experts=8,  # 8 expert networks
    num_experts_per_token=2,  # Activate top-2 per token
    moe_load_balancing=0.01  # Load balancing loss weight
)
model = Forge(config=config)
```

### Benefits
- **Sparse Activation**: Only 2 of 8 experts active per token
- **Scalability**: Add capacity without increasing cost proportionally
- **Specialization**: Each expert learns different patterns

**Note**: Full MoE implementation in FeedForward layer (future PR)

---

## Error Handling

All new features gracefully degrade if dependencies are missing:

```python
try:
    model = Forge.from_huggingface("gpt2")
except ImportError as e:
    print("Install transformers: pip install transformers")

try:
    model.load_lora("adapter.pth")
except ImportError as e:
    print("LoRA support requires lora_utils module")
```

---

## Performance Tips

1. **RoPE Scaling**: Start with `dynamic`, use `yarn` for very long contexts
2. **Speculative Decoding**: Best when draft model is 4-10x smaller
3. **Multi-Modal**: Ensure projection dimensions match encoder outputs
4. **LoRA**: Keep adapters small (rank 8-64 typical)
5. **KV-Cache**: Use quantization for very long contexts

---

## Migration Guide

### From Old Code
```python
# Old (still works!)
model = create_model('small')
```

### To New Features
```python
# New with extended context
config = ForgeConfig(
    vocab_size=8000,
    dim=512,
    n_layers=8,
    n_heads=8,
    max_seq_len=8192,  # Extended!
    rope_scaling_type="dynamic",
    rope_scaling_factor=4.0
)
model = Forge(config=config)
```

No breaking changes - all existing code works unchanged!

---

## Future Enhancements

Coming in future updates:
- Full MoE expert routing implementation
- Tensor parallelism for multi-GPU
- Continuous batching for serving
- AWQ and GPTQ quantization
- More model format loaders

---

## Support

For issues or questions:
1. Check error messages (they guide to solutions)
2. Install optional dependencies as needed
3. See tests in `tests/test_universal_model.py`
4. Refer to inline documentation in `model.py`
