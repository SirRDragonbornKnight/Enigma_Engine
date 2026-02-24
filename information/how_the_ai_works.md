# How the AI Works

Enigma Engine is a **fully local** AI system. Everything runs on your
computer — no cloud, no internet required for inference.

---

## Model Architecture

The engine supports several model formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| GGUF | .gguf | Quantized models (most common, smallest) |
| PyTorch | .pth/.pt | Native PyTorch checkpoints |
| SafeTensors | .safetensors | Safe serialized tensors |
| HuggingFace | folder | Full HuggingFace model directories |

### Transformer Stack

Every model uses the same core transformer architecture:

1. **Token Embedding** — converts text tokens into vectors
2. **Rotary Position Encoding (RoPE)** — encodes token positions
3. **Transformer Blocks** — self-attention + feed-forward layers
4. **RMSNorm** — normalization before each sub-layer
5. **SwiGLU Activation** — gated feed-forward network
6. **Grouped Query Attention (GQA)** — efficient multi-head attention
7. **KV-Cache** — speeds up generation by caching key/value pairs

### Generation

When you send a message, the engine:

1. Tokenizes your text into integer tokens
2. Runs a forward pass through the transformer
3. Samples the next token from the output distribution
4. Repeats until a stop token or max length is reached
5. Decodes the tokens back into text

### Sampling Parameters

| Parameter | What It Does |
|-----------|-------------|
| Temperature | Higher = more creative, lower = more focused |
| Top-P | Nucleus sampling — only consider tokens above this probability |
| Top-K | Only consider the top K most likely tokens |
| Repetition Penalty | Penalizes repeating the same tokens |
| Max Tokens | Maximum number of tokens to generate |

---

## Hardware Detection

The engine auto-detects your hardware:

- **GPU** — CUDA GPUs get automatic layer offloading
- **VRAM** — determines how many layers fit on the GPU
- **RAM** — fallback for CPU-only inference
- **Context Size** — 16K for 24GB+ VRAM, 8K for 12GB+, 4K otherwise

---

## Chat Templates

Models use chat templates to format conversations:

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
<|im_start|>assistant
```

The engine auto-detects the correct template for each model format.
