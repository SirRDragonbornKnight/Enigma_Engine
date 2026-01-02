# Enigma Module System Guide

**Version:** 2.0  
**Last Updated:** December 2024

## Table of Contents

1. [Overview](#overview)
2. [Why Modules?](#why-modules)
3. [Getting Started](#getting-started)
4. [Core Concepts](#core-concepts)
5. [Module Categories](#module-categories)
6. [Conflict Prevention](#conflict-prevention)
7. [Creating Custom Modules](#creating-custom-modules)
8. [Examples](#examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Enigma Module System is a **unified architecture where EVERYTHING is a toggleable module**. This design:

- ✅ **Prevents conflicts** - Only one image generator, one code generator, etc. at a time
- ✅ **Manages dependencies** - Automatically loads required modules
- ✅ **Saves resources** - Unload unused features to free RAM/VRAM
- ✅ **Scales hardware** - Run on Raspberry Pi to datacenter
- ✅ **Simplifies configuration** - One place to control everything

**Philosophy:** Instead of a monolithic system with everything enabled, modules let you compose exactly the AI you need.

---

## Why Modules?

### The Problem

Traditional AI frameworks often have issues:

```python
# ❌ Old way - everything always loaded
engine = AIEngine()  # Loads EVERYTHING
# Uses 16GB RAM even if you only want text chat
```

### The Solution

```python
# ✅ Module way - load only what you need
from enigma.modules import ModuleManager

manager = ModuleManager()
manager.load('model')      # Core AI
manager.load('inference')  # Text generation
# Uses 2GB RAM - perfect for Raspberry Pi!

# Want images? Add it
manager.load('image_gen_local')  # Now uses 8GB RAM
```

### Key Benefits

1. **No Resource Waste** - Don't load image generation if you only need chat
2. **Conflict Prevention** - Can't accidentally load two competing image generators
3. **Clear Dependencies** - Know exactly what each feature needs
4. **Easy Swapping** - Switch from local to cloud generation with one line

---

## Getting Started

### Quick Start (5 lines)

```python
from enigma.modules import ModuleManager

manager = ModuleManager()
manager.load('model', {'size': 'small'})
manager.load('tokenizer')
manager.load('inference')

# Now you can chat!
engine = manager.get_interface('inference')
response = engine.generate("Hello!")
```

### With Generation Capabilities

```python
from enigma.modules import ModuleManager

manager = ModuleManager()

# Core AI
manager.load('model', {'size': 'medium'})
manager.load('tokenizer')
manager.load('inference')

# Add local image generation
manager.load('image_gen_local', {
    'model': 'sd-2.1',
    'steps': 30
})

# Use it
img_gen = manager.get_interface('image_gen_local')
result = img_gen.generate("a sunset over mountains")
```

### Checking Before Loading

```python
can_load, reason = manager.can_load('image_gen_local')
if can_load:
    manager.load('image_gen_local')
else:
    print(f"Cannot load: {reason}")
    # Maybe load cloud version instead
    manager.load('image_gen_api', {'api_key': 'xxx'})
```

---

## Core Concepts

### Module Lifecycle

```
UNLOADED → LOADING → LOADED → ACTIVE → LOADED → UNLOADED
   ↓          ↓         ↓        ↓        ↓         ↓
Create    Load()   Ready    Start   Pause    Cleanup
```

### Module States

- **UNLOADED** - Module exists but not in memory
- **LOADING** - Currently initializing
- **LOADED** - Ready but not actively processing
- **ACTIVE** - Running and processing
- **ERROR** - Failed to load
- **DISABLED** - Administratively disabled

### Module Components

```python
class MyModule(Module):
    INFO = ModuleInfo(
        id="my_module",                    # Unique identifier
        name="My Module",                  # Display name
        description="What it does",        # User-facing description
        category=ModuleCategory.EXTENSION, # Where it belongs
        
        # Dependencies
        requires=["model", "tokenizer"],   # Must load first
        optional=["memory"],               # Nice to have
        conflicts=["other_module"],        # Can't load together
        
        # Capabilities
        provides=["my_capability"],        # What it adds
        
        # Hardware needs
        min_ram_mb=512,
        min_vram_mb=0,
        requires_gpu=False,
    )
    
    def load(self) -> bool:
        # Initialize resources
        return True
    
    def unload(self) -> bool:
        # Clean up
        return True
```

---

## Module Categories

### CORE - Essential AI Components

```python
manager.load('model')        # Transformer model
manager.load('tokenizer')    # Text <-> tokens
manager.load('training')     # Train models
manager.load('inference')    # Generate text
```

**Use when:** Building any AI system

### GENERATION - AI Creation Capabilities

```python
# Images
manager.load('image_gen_local')   # Stable Diffusion (local)
manager.load('image_gen_api')     # DALL-E (cloud)

# Code
manager.load('code_gen_local')    # Enigma model
manager.load('code_gen_api')      # GPT-4

# Video
manager.load('video_gen_local')   # AnimateDiff
manager.load('video_gen_api')     # Replicate

# Audio
manager.load('audio_gen_local')   # pyttsx3
manager.load('audio_gen_api')     # ElevenLabs
```

**Use when:** Adding multimodal AI capabilities

### MEMORY - Storage & Retrieval

```python
manager.load('memory')            # Conversation storage
manager.load('embedding_local')   # Vector search (local)
manager.load('embedding_api')     # OpenAI embeddings
```

**Use when:** Need persistent memory or semantic search

### PERCEPTION - Input Modalities

```python
manager.load('voice_input')  # Speech-to-text
manager.load('vision')       # Screen capture, OCR
```

**Use when:** Adding voice or visual input

### OUTPUT - Presentation

```python
manager.load('voice_output')  # Text-to-speech
manager.load('avatar')        # Visual character
```

**Use when:** Making AI more interactive

### TOOLS - AI Capabilities

```python
manager.load('web_tools')   # Search, fetch pages
manager.load('file_tools')  # File operations
```

**Use when:** AI needs to interact with external systems

### NETWORK - Multi-Device

```python
manager.load('api_server')  # REST API
manager.load('network')     # Device mesh
```

**Use when:** Running AI across multiple devices

### INTERFACE - User Interaction

```python
manager.load('gui')  # Desktop interface
```

**Use when:** Need graphical control panel

---

## Conflict Prevention

### Capability Conflicts

The module system prevents loading modules that provide the same capability:

```python
manager.load('image_gen_local')
# ✅ Success

manager.load('image_gen_api')
# ❌ Error: "Capability 'image_generation' already provided by 'image_gen_local'"
```

**Why?** Two image generators would:
- Waste memory
- Cause confusion about which is used
- Risk generating with wrong model

### Explicit Conflicts

Some modules explicitly conflict:

```python
class ModuleA(Module):
    INFO = ModuleInfo(
        id="module_a",
        conflicts=["module_b"],  # Can't run together
        ...
    )
```

### Dependency Conflicts

Modules check their dependencies before loading:

```python
manager.load('inference')
# ❌ Error: "Required module 'model' not loaded"

manager.load('model')
manager.load('inference')
# ✅ Success - dependency satisfied
```

### Resolution Strategies

**1. Choose One**
```python
# Local or cloud, not both
if has_gpu:
    manager.load('image_gen_local')
else:
    manager.load('image_gen_api', {'api_key': key})
```

**2. Unload First**
```python
manager.unload('image_gen_local')
manager.load('image_gen_api', {'api_key': key})
```

**3. Check Before Load**
```python
can_load, reason = manager.can_load('image_gen_api')
if not can_load:
    if 'image_generation' in reason:
        print("Image gen already loaded")
```

---

## Creating Custom Modules

### Simple Module

```python
from enigma.modules import Module, ModuleInfo, ModuleCategory

class MyModule(Module):
    INFO = ModuleInfo(
        id="my_module",
        name="My Custom Module",
        description="Does something awesome",
        category=ModuleCategory.EXTENSION,
        provides=["my_feature"],
    )
    
    def load(self) -> bool:
        print("Loading my module...")
        self._instance = {"status": "ready"}
        return True
    
    def unload(self) -> bool:
        print("Unloading my module...")
        self._instance = None
        return True
    
    def get_interface(self):
        return self._instance

# Register it
from enigma.modules import ModuleManager

manager = ModuleManager()
manager.register(MyModule)
manager.load('my_module')
```

### Module with Dependencies

```python
class AdvancedModule(Module):
    INFO = ModuleInfo(
        id="advanced",
        name="Advanced Module",
        description="Builds on core AI",
        category=ModuleCategory.EXTENSION,
        requires=["model", "inference"],  # Must load these first
        provides=["advanced_feature"],
    )
    
    def load(self) -> bool:
        # Get other modules
        model = self.manager.get_interface('model')
        inference = self.manager.get_interface('inference')
        
        if not model or not inference:
            return False
        
        self._model = model
        self._inference = inference
        return True
```

### Generation Module (Using Tab Providers)

```python
from enigma.modules.registry import GenerationModule, ModuleInfo, ModuleCategory

class MyImageGenModule(GenerationModule):
    """Custom image generation module using tab providers."""
    
    INFO = ModuleInfo(
        id="my_image_gen",
        name="My Image Generator",
        description="Custom image generation",
        category=ModuleCategory.GENERATION,
        provides=["image_generation"],  # Claims capability
    )
    
    def load(self) -> bool:
        # Import from the tab where implementations live
        from enigma.gui.tabs.image_tab import StableDiffusionLocal
        self._provider = StableDiffusionLocal()
        return self._provider.load()
    
    def generate(self, prompt: str, **kwargs):
        return self._provider.generate(prompt, **kwargs)
```

---

## Examples

### Example 1: Minimal Chat Bot

```python
from enigma.modules import ModuleManager

def create_chatbot():
    manager = ModuleManager()
    
    # Load only what's needed
    manager.load('model', {'size': 'small'})
    manager.load('tokenizer')
    manager.load('inference')
    
    return manager

manager = create_chatbot()
engine = manager.get_interface('inference')

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    response = engine.generate(user_input)
    print(f"AI: {response}")

# Clean up
manager.unload('inference')
manager.unload('tokenizer')
manager.unload('model')
```

### Example 2: Multimodal AI (Text + Images)

```python
from enigma.modules import ModuleManager

manager = ModuleManager()

# Core text AI
manager.load('model', {'size': 'medium'})
manager.load('tokenizer')
manager.load('inference')
manager.load('memory')  # Remember conversations

# Add image generation
if manager.hardware_profile['gpu_available']:
    manager.load('image_gen_local')
else:
    api_key = input("Enter OpenAI API key: ")
    manager.load('image_gen_api', {'api_key': api_key})

# Use both
text_engine = manager.get_interface('inference')
image_gen = manager.get_interface('image_gen_local' if 'image_gen_local' in manager.list_loaded() else 'image_gen_api')

def chat_with_images(prompt):
    # Generate text response
    text = text_engine.generate(prompt)
    
    # If prompt asks for image, generate it
    if "draw" in prompt.lower() or "image" in prompt.lower():
        image = image_gen.generate(prompt)
        return {"text": text, "image": image}
    
    return {"text": text}
```

### Example 3: Hardware-Aware Loading

```python
from enigma.modules import ModuleManager

def smart_load(manager):
    """Load modules based on available hardware."""
    hw = manager.hardware_profile
    
    # Always load core
    manager.load('model', {'size': 'small'})
    manager.load('tokenizer')
    manager.load('inference')
    
    # Memory if we have enough RAM
    if hw['ram_mb'] >= 4096:
        manager.load('memory')
    
    # GPU features
    if hw['gpu_available'] and hw['vram_mb'] >= 6000:
        manager.load('image_gen_local')
    
    # Voice on any system
    manager.load('voice_output')
    
    print(f"Loaded: {manager.list_loaded()}")

manager = ModuleManager()
smart_load(manager)
```

### Example 4: Dynamic Module Switching

```python
from enigma.modules import ModuleManager

manager = ModuleManager()
manager.load('model')
manager.load('tokenizer')
manager.load('inference')

def switch_image_provider(use_cloud=False):
    """Switch between local and cloud image generation."""
    
    # Unload current if any
    if 'image_gen_local' in manager.list_loaded():
        manager.unload('image_gen_local')
    if 'image_gen_api' in manager.list_loaded():
        manager.unload('image_gen_api')
    
    # Load new one
    if use_cloud:
        api_key = input("API key: ")
        manager.load('image_gen_api', {'api_key': api_key})
        print("Switched to cloud generation")
    else:
        manager.load('image_gen_local')
        print("Switched to local generation")

# Start with local
switch_image_provider(use_cloud=False)

# User says it's too slow, switch to cloud
switch_image_provider(use_cloud=True)
```

---

## Best Practices

### 1. Always Check Hardware

```python
hw = manager.hardware_profile

if hw['gpu_available']:
    # Use GPU modules
    manager.load('image_gen_local')
else:
    # Fall back to CPU/cloud
    manager.load('image_gen_api', config)
```

### 2. Handle Load Failures

```python
success = manager.load('optional_module')
if not success:
    print("Module failed to load, continuing without it")
    # Don't crash, adapt
```

### 3. Use can_load()

```python
can_load, reason = manager.can_load('module_id')
if can_load:
    manager.load('module_id')
else:
    print(f"Can't load: {reason}")
    # Show user why, suggest alternatives
```

### 4. Clean Up Resources

```python
try:
    # Use modules
    engine = manager.get_interface('inference')
    result = engine.generate("test")
finally:
    # Always clean up
    manager.unload('inference')
```

### 5. Save/Load Configuration

```python
# Save which modules are loaded
manager.save_config(Path("my_config.json"))

# Later, restore
new_manager = ModuleManager()
new_manager.load_config(Path("my_config.json"))
```

### 6. Register Early

```python
# At startup, register all modules once
from enigma.modules.registry import register_all

manager = ModuleManager()
register_all(manager)

# Now load as needed throughout app
```

### 7. Use Categories to Organize

```python
from enigma.modules import ModuleCategory

# Load all core modules
for module_info in manager.list_modules(ModuleCategory.CORE):
    if manager.can_load(module_info.id)[0]:
        manager.load(module_info.id)
```

---

## Troubleshooting

### "Module not registered"

**Problem:** Trying to load a module that wasn't registered.

**Solution:**
```python
from enigma.modules.registry import register_all

manager = ModuleManager()
register_all(manager)  # Register built-in modules
```

### "Required module not loaded"

**Problem:** Loading a module before its dependencies.

**Solution:**
```python
# Load in correct order
manager.load('model')       # ← Load this first
manager.load('tokenizer')   # ← Then this
manager.load('inference')   # ← Then this (needs model + tokenizer)
```

### "Capability already provided"

**Problem:** Trying to load two modules that do the same thing.

**Solution:**
```python
# Unload one first
manager.unload('image_gen_local')
manager.load('image_gen_api', config)
```

### "Module requires GPU but none available"

**Problem:** GPU-only module on CPU-only system.

**Solution:**
```python
if manager.hardware_profile['gpu_available']:
    manager.load('image_gen_local')
else:
    manager.load('image_gen_api', {'api_key': key})
```

### Module Loads But Doesn't Work

**Problem:** Module loaded successfully but interface is None.

**Solution:**
```python
# Check module state
module = manager.get_module('my_module')
print(f"State: {module.state}")

# Check interface
interface = manager.get_interface('my_module')
if interface is None:
    print("Module didn't set _instance in load()")
```

---

## Advanced Topics

### Custom Module Discovery

```python
# Auto-discover modules in a directory
import importlib
from pathlib import Path

def discover_modules(path: Path):
    for py_file in path.glob("*_module.py"):
        module_name = py_file.stem
        mod = importlib.import_module(f"my_modules.{module_name}")
        
        # Find Module subclasses
        for item in dir(mod):
            obj = getattr(mod, item)
            if isinstance(obj, type) and issubclass(obj, Module) and obj != Module:
                manager.register(obj)
```

### Module Hot-Reloading

```python
def reload_module(module_id: str):
    """Reload a module without restarting."""
    config = manager.get_module(module_id).config
    manager.unload(module_id)
    manager.load(module_id, config)
```

### Conditional Loading Based on Task

```python
TASK_MODULES = {
    'chat': ['model', 'tokenizer', 'inference', 'memory'],
    'image': ['model', 'tokenizer', 'inference', 'image_gen_local'],
    'code': ['model', 'tokenizer', 'inference', 'code_gen_local'],
}

def setup_for_task(task: str):
    for module_id in TASK_MODULES.get(task, []):
        if manager.can_load(module_id)[0]:
            manager.load(module_id)
```

---

## Conclusion

The Enigma Module System provides:

✅ **Flexibility** - Load only what you need  
✅ **Safety** - Automatic conflict prevention  
✅ **Clarity** - Explicit dependencies and capabilities  
✅ **Efficiency** - Minimal resource usage  
✅ **Scalability** - Raspberry Pi to datacenter  

**Next Steps:**
1. Try the [Quick Start](#getting-started) example
2. Browse available modules in `enigma.modules.registry`
3. Create your first custom module
4. Join the community and share your modules!

---

**Questions?** Open an issue on GitHub or check the docs folder for more guides.
