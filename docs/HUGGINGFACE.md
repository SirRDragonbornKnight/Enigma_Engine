# HuggingFace Integration

## Overview

Enigma supports HuggingFace models through the GUI tabs and module system for:
- Text generation (via Code Tab or chat)
- Image generation (via Image Tab using Stable Diffusion)
- Text embeddings (via Embeddings Tab using sentence-transformers)
- Text-to-speech (via Audio Tab)

## Installation

```bash
pip install huggingface-hub transformers diffusers sentence-transformers
```

## Usage Through GUI Tabs

### Image Generation (Image Tab)

The Image Tab uses Stable Diffusion models from HuggingFace:

```python
from forge_ai.gui.tabs.image_tab import StableDiffusionLocal

# Create local image generator
generator = StableDiffusionLocal(
    model_id="stabilityai/stable-diffusion-2-1"  # HuggingFace model
)
generator.load()

result = generator.generate(
    prompt="a beautiful sunset",
    width=512,
    height=512
)
print(f"Saved to: {result['path']}")
```

### Embeddings (Embeddings Tab)

The Embeddings Tab uses sentence-transformers from HuggingFace:

```python
from forge_ai.gui.tabs.embeddings_tab import LocalEmbedding

# Create local embedding generator
embedder = LocalEmbedding(
    model_name="all-MiniLM-L6-v2"  # sentence-transformers model
)
embedder.load()

result = embedder.embed("Hello world")
embedding = result["embedding"]  # List of floats

# Compare similarity
similarity = embedder.similarity("Hello", "Hi there")
print(f"Similarity: {similarity['similarity']:.2%}")
```

### Audio/TTS (Audio Tab)

```python
from forge_ai.gui.tabs.audio_tab import LocalTTS

# Create local TTS
tts = LocalTTS()
tts.load()

result = tts.generate("Hello, world!")
print(f"Saved to: {result['path']}")
```

## Using Through Module System

The module system wraps the tab implementations:

```python
from forge_ai.modules import ModuleManager

manager = ModuleManager()

# Load local image generation (Stable Diffusion)
manager.load('image_gen_local')
image_mod = manager.get_module('image_gen_local')
result = image_mod.generate("a sunset")

# Load local embeddings (sentence-transformers)
manager.load('embedding_local')
embed_mod = manager.get_module('embedding_local')
result = embed_mod.embed("Hello world")
```

## Supported Models

### Image Generation (Stable Diffusion)
- `stabilityai/stable-diffusion-2-1` (default)
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-xl-base-1.0`
- `stabilityai/sdxl-turbo`

### Embeddings (sentence-transformers)
- `all-MiniLM-L6-v2` (default, 384 dims)
- `all-mpnet-base-v2` (768 dims)
- Any sentence-transformers model

### 3D Generation (Shap-E)
- `openai/shap-e` (via 3D Tab)

### Video Generation (AnimateDiff)
- `guoyww/animatediff-motion-adapter-v1-5-2` (via Video Tab)

## API Keys

For cloud providers (not HuggingFace specific):

```bash
# OpenAI (for cloud image/code/embeddings)
export OPENAI_API_KEY="your_key"

# Replicate (for cloud video/audio/3D)
export REPLICATE_API_TOKEN="your_key"

# ElevenLabs (for cloud TTS)
export ELEVENLABS_API_KEY="your_key"
```

## Local vs Cloud

**Local Providers** (default, HuggingFace-based):
- `StableDiffusionLocal` - Image generation
- `LocalEmbedding` - Text embeddings
- `LocalVideo` - Video generation (AnimateDiff)
- `Local3DGen` - 3D generation (Shap-E)
- `LocalTTS` - Text-to-speech (pyttsx3)

**Cloud Providers** (require API keys):
- `OpenAIImage`, `ReplicateImage` - Cloud images
- `OpenAIEmbedding` - Cloud embeddings
- `ReplicateVideo` - Cloud video
- `Cloud3DGen` - Cloud 3D
- `ElevenLabsTTS`, `ReplicateAudio` - Cloud audio

## Note on Legacy Addons

The previous addon system (`forge_ai.addons`) has been removed. All functionality is now in the GUI tabs:
- `forge_ai/gui/tabs/image_tab.py`
- `forge_ai/gui/tabs/code_tab.py`
- `forge_ai/gui/tabs/video_tab.py`
- `forge_ai/gui/tabs/audio_tab.py`
- `forge_ai/gui/tabs/embeddings_tab.py`
- `forge_ai/gui/tabs/threed_tab.py`

Each tab contains both the implementation classes and the GUI.
## Exporting ForgeAI Models to HuggingFace

You can upload your locally trained ForgeAI models to HuggingFace Hub to share them!

### Quick Export

```python
from forge_ai.core.model_registry import ModelRegistry

registry = ModelRegistry()

# Export locally (creates HuggingFace-compatible format)
registry.export_to_huggingface("my_model", output_dir="./my_model_hf")

# Push directly to HuggingFace Hub
registry.export_to_huggingface(
    "my_model",
    repo_id="your-username/my-forge-model",
    token="hf_your_token_here",  # Or set HF_TOKEN env var
    private=False  # Make it public
)
```

### Using the Exporter Directly

```python
from forge_ai.core.huggingface_exporter import HuggingFaceExporter

exporter = HuggingFaceExporter()

# List models that can be exported
exportable = exporter.list_exportable_models()
print(exportable)

# Export to local directory
exporter.export_to_hf_format("my_model", output_dir="./export")

# Push to Hub
url = exporter.push_to_hub(
    model_name="my_model",
    repo_id="username/model-name",
    token="hf_...",
    private=False
)
print(f"Model available at: {url}")
```

### Convenience Functions

```python
from forge_ai.core import export_model_to_hub, export_model_locally

# One-liner to push to Hub
url = export_model_to_hub("my_model", "username/my-model", token="hf_...")

# One-liner to export locally
path = export_model_locally("my_model", "./exported")
```

### Requirements

```bash
# For local export
pip install safetensors

# For pushing to Hub
pip install huggingface-hub
```

### Getting a HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Write" permissions
3. Either pass it directly or set as environment variable:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

### What Gets Exported

- `config.json` - Model architecture in HuggingFace format
- `model.safetensors` - Model weights (or `pytorch_model.bin`)
- `tokenizer.json` - Tokenizer files (if available)
- `README.md` - Model card with usage instructions

## Multi-Platform Model Hub (Bidirectional)

ForgeAI has a complete Model Hub for **importing AND exporting** models to multiple platforms!

### Available Platforms

| Platform | Import | Export | Auth Required |
|----------|--------|--------|---------------|
| **HuggingFace** | ✅ Search & download | ✅ Push to Hub | `HF_TOKEN` |
| **Replicate** | ✅ API wrappers | ✅ Cog packages | `REPLICATE_API_TOKEN` |
| **Ollama** | ✅ Pull models | ✅ GGUF export | None |
| **Weights & Biases** | ✅ Download artifacts | ✅ Log artifacts | `WANDB_API_KEY` |
| **ONNX** | ❌ | ✅ Edge deployment | None |

### Using the Model Hub

```python
from forge_ai.core.model_export import ModelHub

hub = ModelHub()

# === IMPORT: Get models FROM platforms ===

# Search for models
models = hub.search("llama", provider="huggingface")
models = hub.search("mistral", provider="ollama")

# Import a model from HuggingFace
result = hub.import_model(
    "microsoft/DialoGPT-small",
    provider="huggingface",
    local_name="dialogpt"
)
print(f"Downloaded to: {result.local_path}")

# Import from Ollama (pulls from ollama.com)
result = hub.import_model(
    "llama2:7b",
    provider="ollama"
)

# Import from Replicate (creates API wrapper for cloud model)
result = hub.import_model(
    "meta/llama-2-70b-chat",
    provider="replicate",
    local_name="llama2-api"
)

# List locally installed Ollama models
models = hub.list_available(provider="ollama")

# === EXPORT: Push models TO platforms ===

# Export to HuggingFace
hub.export("my_model", "huggingface", repo_id="user/model")

# Export to Ollama (GGUF format)
hub.export("my_model", "ollama", quantization="q4_k_m")

# Export to ONNX (for edge/mobile)
hub.export("my_model", "onnx", optimize=True)

# Export to multiple platforms at once
results = hub.export_all(
    "my_model",
    providers=["huggingface", "ollama", "onnx"],
    huggingface_kwargs={"repo_id": "user/model"}
)
```

### Quick Functions

```python
from forge_ai.core.model_export import (
    import_model,
    export_model,
    search_models,
    list_import_providers,
    list_export_providers,
)

# One-liner imports
result = import_model("microsoft/DialoGPT-small", "huggingface", local_name="dialogpt")
result = import_model("llama2:7b", "ollama")

# One-liner exports
export_model("my_model", "huggingface", repo_id="user/model")
export_model("my_model", "ollama")

# Search for models
models = search_models("llama", "huggingface")
models = search_models("mistral", "ollama")

# List available providers
print(list_import_providers())  # ['huggingface', 'replicate', 'ollama', 'wandb']
print(list_export_providers())  # ['huggingface', 'replicate', 'ollama', 'wandb', 'onnx']
```

### Platform-Specific Details

#### HuggingFace
- **Import**: Downloads model weights, config, tokenizer
- **Export**: Pushes safetensors/pytorch_model.bin with model card
- Token: https://huggingface.co/settings/tokens (needs "Write" for export)

#### Ollama
- **Import**: Runs `ollama pull` to download GGUF models
- **Export**: Converts to GGUF, creates Modelfile, registers locally
- No auth needed - works with local ollama server

#### Replicate
- **Import**: Creates an API wrapper for cloud models (no download)
- **Export**: Creates Cog package structure for deployment
- Token: https://replicate.com/account

#### Weights & Biases
- **Import**: Downloads model artifacts from W&B projects
- **Export**: Logs model as artifact to W&B
- Token: https://wandb.ai/authorize

See the individual provider documentation for more details on each platform.