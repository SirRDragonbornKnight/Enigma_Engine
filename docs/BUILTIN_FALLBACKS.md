# ForgeAI Built-in Fallbacks

ForgeAI now includes **zero-dependency fallbacks** that work without installing external packages like `pyttsx3`, `sentence-transformers`, `diffusers`, etc.

## How It Works

When a full-featured library isn't available, ForgeAI automatically falls back to built-in implementations:

| Feature | Full Library | Built-in Fallback |
|---------|-------------|-------------------|
| **TTS (Speech)** | pyttsx3 | Windows SAPI / macOS `say` / Linux espeak |
| **Embeddings** | sentence-transformers | TF-IDF + hash-based vectors |
| **Code Generation** | ForgeAI Model | Template-based code snippets |
| **Image Generation** | Stable Diffusion | Procedural art (gradients, patterns, etc.) |

## Usage

The fallbacks are **automatic**. When you use any tab in the GUI, it will:
1. Try to load the full-featured library
2. If that fails, silently fall back to the built-in version
3. Print a message like "Using built-in TTS (system speech)"

## Built-in Modules

Located in `forge_ai/builtin/`:

### BuiltinTTS (`tts.py`)
- Uses system speech APIs (no installation needed)
- Windows: PowerShell SAPI
- macOS: `say` command
- Linux: `espeak` command
- Supports voice selection, rate, volume

### BuiltinEmbeddings (`embeddings.py`)
- Pure Python text vectorization
- TF-IDF inspired word weighting
- Character n-grams for subword features
- 384-dimensional output (same as MiniLM)
- ~10x slower than sentence-transformers but works anywhere

### BuiltinCodeGen (`code_gen.py`)
- Template-based code generation
- Supports Python, JavaScript, HTML, SQL, Bash
- Generates functions, classes, APIs, tests
- Not AI-powered but useful for scaffolding

### BuiltinImageGen (`image_gen.py`)
- Pure Python PNG generation
- Procedural art styles:
  - Gradients
  - Geometric patterns
  - Stars/space
  - Waves
  - Terrain/landscapes
  - Abstract art
- Analyzes prompt for colors and style hints

## Direct Usage

```python
from forge_ai.builtin import (
    BuiltinTTS,
    BuiltinEmbeddings,
    BuiltinCodeGen,
    BuiltinImageGen,
)

# Text-to-speech
tts = BuiltinTTS()
tts.load()
tts.speak("Hello from ForgeAI!")

# Embeddings
emb = BuiltinEmbeddings()
emb.load()
result = emb.embed("Hello world")
print(f"Vector dimensions: {result['dimensions']}")

# Code generation
code = BuiltinCodeGen()
code.load()
result = code.generate("create a login function", language="python")
print(result["code"])

# Image generation
img = BuiltinImageGen(512, 512)
img.load()
result = img.generate("a sunset over mountains")
with open("sunset.png", "wb") as f:
    f.write(result["image_data"])
```

## Quality Trade-offs

| Feature | Full Version | Built-in |
|---------|-------------|----------|
| TTS Quality | Natural voices | Robotic system voice |
| Embedding Quality | Semantic similarity | Word overlap similarity |
| Code Quality | AI-generated logic | Template scaffolds |
| Image Quality | AI art | Simple procedural art |

## When to Install Full Libraries

Install the full libraries when you need:

- **pyttsx3**: Natural-sounding offline speech
- **sentence-transformers**: Semantic search, similarity
- **diffusers + torch**: AI image generation
- **openai**: Cloud AI capabilities

```bash
# Minimal (built-ins only)
pip install PyQt5

# Full local features
pip install pyttsx3 sentence-transformers diffusers torch

# Cloud features
pip install openai replicate elevenlabs
```

## Checking Status

```python
from forge_ai.builtin import get_builtin_status

status = get_builtin_status()
print(status)
# {'tts': True, 'embeddings': True, 'code_gen': True, 'image_gen': True}
```

All built-ins are always available (they use only Python standard library + system commands).
