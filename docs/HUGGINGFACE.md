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
from enigma.gui.tabs.image_tab import StableDiffusionLocal

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
from enigma.gui.tabs.embeddings_tab import LocalEmbedding

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
from enigma.gui.tabs.audio_tab import LocalTTS

# Create local TTS
tts = LocalTTS()
tts.load()

result = tts.generate("Hello, world!")
print(f"Saved to: {result['path']}")
```

## Using Through Module System

The module system wraps the tab implementations:

```python
from enigma.modules import ModuleManager

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

The previous addon system (`enigma.addons`) has been removed. All functionality is now in the GUI tabs:
- `enigma/gui/tabs/image_tab.py`
- `enigma/gui/tabs/code_tab.py`
- `enigma/gui/tabs/video_tab.py`
- `enigma/gui/tabs/audio_tab.py`
- `enigma/gui/tabs/embeddings_tab.py`
- `enigma/gui/tabs/threed_tab.py`

Each tab contains both the implementation classes and the GUI.
