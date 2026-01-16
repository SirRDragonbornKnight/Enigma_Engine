# Local vs Cloud Modules

ForgeAI is designed to be an **"open black box"** - completely transparent about what runs locally vs what connects to external services.

## � LOCAL-FIRST DESIGN

**ForgeAI prioritizes local execution by default.** All generation tabs (image, code, video, audio, embeddings, 3D) will automatically use local providers. Cloud providers are only used if explicitly configured.

This means:
- **Zero data sent to cloud services by default**
- **Works offline out of the box**
- **No API keys required for basic functionality**
- **Your data stays on your machine**

## �🏠 100% Local Modules (No Internet Required)

These modules run entirely on your machine with **zero external API calls**:

### Core Modules
- **model** - Your Enigma transformer model (RoPE, RMSNorm, SwiGLU, GQA)
- **tokenizer** - Text tokenization (BPE, character, custom vocabularies)
- **training** - Model training system
- **inference** - Text generation engine

### Generation (Local)
- **image_gen_local** - Stable Diffusion (requires GPU with 8GB+ VRAM)
- **code_gen_local** - Uses your trained Enigma model for code
- **video_gen_local** - AnimateDiff for video (requires powerful GPU)
- **audio_gen_local** - pyttsx3 or edge-tts for text-to-speech

### Memory & Storage
- **memory** - Conversation storage (JSON/SQLite)
- **embedding_local** - sentence-transformers for semantic search

### Perception & Output
- **voice_input** - Speech-to-text (local engines)
- **voice_output** - Text-to-speech (pyttsx3, edge-tts)
- **vision** - Image capture, OCR, object detection
- **avatar** - Visual AI representation

### Tools & Interface
- **web_tools** - Web search and page fetching (requires internet for web access, but no cloud APIs)
- **file_tools** - File system operations
- **gui** - PyQt5 graphical interface
- **api_server** - Local REST API server
- **network** - Multi-device communication (local network)

## ☁️ Cloud/API Modules (Require API Keys + Internet)

These modules connect to external cloud services:

### Generation (Cloud)
- **image_gen_api** - OpenAI DALL-E or Replicate APIs
  - Providers: OpenAI, Replicate
  - Requires: API key, internet connection
  - Data sent: Image prompts and parameters
  
- **code_gen_api** - OpenAI GPT-4 for code generation
  - Provider: OpenAI
  - Requires: API key, internet connection
  - Data sent: Code prompts and context
  
- **video_gen_api** - Replicate video generation
  - Provider: Replicate
  - Requires: API token, internet connection
  - Data sent: Video prompts and parameters
  
- **audio_gen_api** - ElevenLabs TTS or Replicate MusicGen
  - Providers: ElevenLabs, Replicate
  - Requires: API key, internet connection
  - Data sent: Text for speech synthesis or music prompts

### Embeddings (Cloud)
- **embedding_api** - OpenAI embeddings
  - Provider: OpenAI
  - Requires: API key, internet connection
  - Data sent: Text for embedding generation

## 🔒 Local-Only Mode (Default)

**By default, ForgeAI runs in local-only mode** for maximum privacy.

### What This Means
- Cloud modules **cannot** be loaded
- No API keys required
- No data leaves your machine
- 100% offline capable (except web_tools for web browsing)

### Enabling Cloud Modules

To use cloud services, you must explicitly disable local-only mode:

```python
from forge_ai.modules import ModuleManager

# Create manager with cloud modules enabled
manager = ModuleManager(local_only=False)

# Now you can load cloud modules
manager.load('image_gen_api', config={'api_key': 'your-key-here'})
```

### Warning System
When loading cloud modules, you'll see a warning:

```
⚠️  Warning: Module 'image_gen_api' connects to external cloud services 
    and requires API keys + internet.
```

## 📋 Quick Reference

### Command: List Local-Only Modules
```python
from forge_ai.modules import registry

# Get all local modules
local_modules = registry.list_local_modules()
for module in local_modules:
    print(f"{module.id}: {module.name}")
```

### Command: List Cloud Modules
```python
from forge_ai.modules import registry

# Get all cloud modules
cloud_modules = registry.list_cloud_modules()
for module in cloud_modules:
    print(f"{module.id}: {module.name} - {module.description}")
```

### Check If Module Is Local
```python
from forge_ai.modules import registry

module_class = registry.get_module('image_gen_local')
info = module_class.get_info()
print(f"Is cloud service: {info.is_cloud_service}")  # False
```

## 🎯 Best Practices

### For Maximum Privacy
1. Keep `local_only=True` (default)
2. Only use `*_local` modules
3. Review module descriptions before loading
4. Check `is_cloud_service` flag in ModuleInfo

### For Best Quality (with cloud)
1. Set `local_only=False` when creating ModuleManager
2. Use cloud modules for specific high-quality tasks
3. Use local modules for everything else
4. Be aware of API costs

### Recommended Configurations

#### Raspberry Pi / Low-End PC (100% Local)
```python
manager = ModuleManager(local_only=True)
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('memory')
manager.load('audio_gen_local')  # pyttsx3 TTS
```

#### Desktop PC with GPU (100% Local + Generation)
```python
manager = ModuleManager(local_only=True)
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('image_gen_local')  # Stable Diffusion
manager.load('audio_gen_local')
manager.load('embedding_local')
```

#### Hybrid (Local + Cloud for specific tasks)
```python
manager = ModuleManager(local_only=False)
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('image_gen_local')  # Primary: local SD
manager.load('code_gen_api', config={'api_key': 'sk-...'})  # Complex code only
```

## ❓ FAQ

**Q: Can I use Enigma without any internet connection?**  
A: Yes! Just use local-only mode (default) and stick to `*_local` modules.

**Q: Do cloud modules send my conversations to external servers?**  
A: Cloud generation modules only send the specific prompts you give them (e.g., "generate an image of a cat"). Your conversation history in the `memory` module stays local.

**Q: Which modules are best for privacy?**  
A: All `*_local` modules and core modules. They never make external API calls.

**Q: Are cloud modules expensive?**  
A: Costs vary by provider:
- OpenAI DALL-E: ~$0.04 per image
- OpenAI GPT-4: ~$0.03 per 1K tokens
- ElevenLabs: ~$0.30 per 1K characters
- Replicate: Varies by model

**Q: Can I switch between local and cloud versions?**  
A: Yes, but only one can be loaded at a time (they both provide the same capability). Unload one before loading the other.

**Q: What about web_tools module?**  
A: `web_tools` accesses the internet for web browsing but doesn't send data to third-party APIs. It's marked as local since it doesn't require API keys or connect to specific cloud services.

## 🔐 Security & Privacy Summary

| Category | Local Modules | Cloud Modules |
|----------|---------------|---------------|
| Internet Required | ❌ No | ✅ Yes |
| API Keys Required | ❌ No | ✅ Yes |
| Data Leaves Machine | ❌ Never | ✅ Yes (prompts only) |
| Costs Money | ❌ Free | ✅ Pay per use |
| Works Offline | ✅ Yes | ❌ No |
| Privacy | 🔒 100% Private | ⚠️ Depends on provider |

---

**Remember**: ForgeAI's default is **local-only** for your privacy and security. Cloud modules require explicit opt-in.
