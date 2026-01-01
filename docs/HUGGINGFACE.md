# HuggingFace Integration

## Overview

Enigma supports HuggingFace models for:
- Text generation
- Image generation
- Text embeddings
- Text-to-speech

## Installation

```bash
pip install huggingface-hub transformers
```

## Usage

### Text Generation

```python
from enigma.addons.huggingface import HuggingFaceTextGeneration

# Use HuggingFace Inference API
addon = HuggingFaceTextGeneration(
    model_name="gpt2",
    api_key="your_hf_token"  # or set HUGGINGFACE_TOKEN env var
)
addon.load()

result = addon.generate(
    prompt="Tell me a story",
    max_tokens=200,
    temperature=0.7
)
print(result.data)

# Or use local model
addon_local = HuggingFaceTextGeneration(
    model_name="gpt2",
    use_local=True
)
```

### Image Generation

```python
from enigma.addons.huggingface import HuggingFaceImageGeneration

addon = HuggingFaceImageGeneration(
    model_name="stabilityai/stable-diffusion-2-1",
    use_local=True  # Download and run locally
)
addon.load()

result = addon.generate(
    prompt="a beautiful sunset",
    width=512,
    height=512
)

# Save image
with open("output.png", "wb") as f:
    f.write(result.data)
```

### Embeddings

```python
from enigma.addons.huggingface import HuggingFaceEmbeddings

addon = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_local=True
)
addon.load()

result = addon.generate("Hello world")
embedding = result.data  # List of floats
```

### Text-to-Speech

```python
from enigma.addons.huggingface import HuggingFaceTTS

addon = HuggingFaceTTS(
    model_name="facebook/mms-tts-eng",
    use_local=True
)
addon.load()

result = addon.text_to_speech("Hello, world!")
```

## Using with Addon Registry

```python
from enigma.addons.builtin import BUILTIN_ADDONS

# Check if available
if 'huggingface_text' in BUILTIN_ADDONS:
    addon_class = BUILTIN_ADDONS['huggingface_text']
    addon = addon_class(model_name="gpt2")
    addon.load()
```

## Supported Models

### Text Generation
- gpt2, gpt2-medium, gpt2-large
- meta-llama/Llama-2-7b-chat-hf (requires authentication)
- bigscience/bloom-560m
- Any HuggingFace causal LM model

### Image Generation
- stabilityai/stable-diffusion-2-1
- runwayml/stable-diffusion-v1-5
- CompVis/stable-diffusion-v1-4

### Embeddings
- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2

### TTS
- facebook/mms-tts-eng
- facebook/fastspeech2-en-ljspeech

## API Keys

Set your HuggingFace token:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

Or pass directly:

```python
addon = HuggingFaceTextGeneration(
    model_name="gpt2",
    api_key="your_token"
)
```

## Local vs API

**Local (use_local=True)**:
- Pros: Free, private, no internet needed
- Cons: Requires disk space, slower on CPU

**API (use_local=False)**:
- Pros: Fast, no local resources needed
- Cons: Requires API key, internet, may have rate limits

## See Also

- [Addon System](../docs/MODULE_GUIDE.md)
- [HuggingFace Models](https://huggingface.co/models)
