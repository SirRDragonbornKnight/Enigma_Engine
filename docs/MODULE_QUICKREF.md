# Module System Quick Reference

**One-page reference for the Enigma Module System**

## Basic Usage

```python
from enigma.modules import ModuleManager
from enigma.modules.registry import register_all

# Create manager
manager = ModuleManager()
register_all(manager)  # Load all built-in modules

# Load modules
manager.load('model', {'size': 'small'})
manager.load('tokenizer')
manager.load('inference')

# Use modules
engine = manager.get_interface('inference')
response = engine.generate("Hello!")

# Unload
manager.unload('inference')
```

## Available Modules (24 total)

### Core (4)
```
model        - Transformer AI model
tokenizer    - Text tokenization  
training     - Train models
inference    - Generate text
```

### Generation (8)
```
image_gen_local  - Stable Diffusion (GPU)
image_gen_api    - DALL-E / Replicate (cloud)
code_gen_local   - Enigma code generation
code_gen_api     - GPT-4 code generation
video_gen_local  - AnimateDiff (GPU)
video_gen_api    - Replicate videos
audio_gen_local  - pyttsx3 TTS
audio_gen_api    - ElevenLabs TTS
```

### Memory (3)
```
memory            - Conversation storage
embedding_local   - Local vector search
embedding_api     - OpenAI embeddings
```

### Perception (2)
```
voice_input  - Speech-to-text
vision       - Screen capture, OCR
```

### Output (2)
```
voice_output - Text-to-speech
avatar       - Visual character
```

### Tools (2)
```
web_tools   - Web search, fetching
file_tools  - File operations
```

### Network (2)
```
api_server - REST API
network    - Multi-device mesh
```

### Interface (1)
```
gui - Desktop application
```

## Common Patterns

### Check Before Load
```python
can_load, reason = manager.can_load('image_gen_local')
if can_load:
    manager.load('image_gen_local')
else:
    print(f"Can't load: {reason}")
```

### Load with Dependencies
```python
# Load in order
manager.load('model')      # Required by inference
manager.load('tokenizer')  # Required by inference  
manager.load('inference')  # Now works
```

### Swap Modules
```python
# Switch from local to cloud
manager.unload('image_gen_local')
manager.load('image_gen_api', {'api_key': key})
```

### Hardware-Aware Loading
```python
hw = manager.hardware_profile
if hw['gpu_available'] and hw['vram_mb'] >= 6000:
    manager.load('image_gen_local')
else:
    manager.load('image_gen_api', config)
```

### List Loaded
```python
loaded = manager.list_loaded()
print(f"Running: {loaded}")
```

### Get Status
```python
status = manager.get_status()
print(f"RAM: {status['hardware']['ram_mb']} MB")
print(f"Loaded: {status['loaded']} modules")
```

## Module Categories

```python
from enigma.modules import ModuleCategory

# Get all generation modules
gen_modules = manager.list_modules(ModuleCategory.GENERATION)
for mod in gen_modules:
    print(f"{mod.id}: {mod.name}")
```

## Conflict Prevention

**Automatic:** System prevents loading modules that provide same capability

```python
manager.load('image_gen_local')   # ✓ OK
manager.load('image_gen_api')     # ✗ Conflict: both provide 'image_generation'
```

**Solution:** Unload first
```python
manager.unload('image_gen_local')
manager.load('image_gen_api', config)
```

## Configuration

### Load with Config
```python
manager.load('model', {
    'size': 'medium',
    'vocab_size': 10000,
    'device': 'cuda'
})
```

### Update Config
```python
module = manager.get_module('model')
module.configure({'size': 'large'})
```

### Save/Load Config
```python
# Save current state
manager.save_config(Path('config.json'))

# Restore later
new_manager = ModuleManager()
new_manager.load_config(Path('config.json'))
```

## Module States

```
UNLOADED  - Not in memory
LOADING   - Initializing
LOADED    - Ready to use
ACTIVE    - Processing
ERROR     - Failed to load
DISABLED  - Administratively disabled
```

## Creating Custom Modules

### Minimal Module
```python
from enigma.modules import Module, ModuleInfo, ModuleCategory

class MyModule(Module):
    INFO = ModuleInfo(
        id="my_module",
        name="My Module",
        description="Does something",
        category=ModuleCategory.EXTENSION,
    )
    
    def load(self) -> bool:
        self._instance = {"ready": True}
        return True
```

### With Dependencies
```python
INFO = ModuleInfo(
    id="my_module",
    requires=["model", "tokenizer"],  # Must load first
    optional=["memory"],              # Nice to have
    conflicts=["other_module"],       # Can't run together
    provides=["my_capability"],       # What it adds
)
```

### Register & Use
```python
manager.register(MyModule)
manager.load('my_module')
interface = manager.get_interface('my_module')
```

## Common Issues

### "Module not registered"
```python
# Fix: Register first
from enigma.modules.registry import register_all
register_all(manager)
```

### "Required module not loaded"  
```python
# Fix: Load dependencies first
manager.load('model')
manager.load('tokenizer')
manager.load('inference')  # Now works
```

### "Capability already provided"
```python
# Fix: Unload conflicting module
manager.unload('image_gen_local')
manager.load('image_gen_api', config)
```

### "Module requires GPU"
```python
# Fix: Check hardware first
if manager.hardware_profile['gpu_available']:
    manager.load('image_gen_local')
else:
    manager.load('image_gen_api', config)
```

## Resources

- **Full Guide:** `docs/MODULE_GUIDE.md`
- **Examples:** `examples/module_system_demo.py`
- **Tests:** `tests/test_modules.py`
- **Source:** `enigma/modules/`

## Quick Recipes

### Chatbot
```python
manager.load('model', {'size': 'small'})
manager.load('tokenizer')
manager.load('inference')
manager.load('memory')
```

### Image Generator
```python
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('image_gen_local')  # or image_gen_api
```

### Voice Assistant
```python
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('voice_input')
manager.load('voice_output')
```

### Multimodal AI
```python
manager.load('model', {'size': 'large'})
manager.load('tokenizer')
manager.load('inference')
manager.load('memory')
manager.load('vision')
manager.load('image_gen_local')
manager.load('voice_output')
```

---

**For complete documentation, see `docs/MODULE_GUIDE.md`**
