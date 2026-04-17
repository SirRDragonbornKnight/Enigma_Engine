# Mods - Standalone AI Services

Modular AI services that can run independently or connect to a central router.

## Available Services

| Service | Port | Description |
|---------|------|-------------|
| **router** | 9900 | Central hub for service coordination |
| **imagegen** | 9901 | Image generation (SD, DALL-E, Replicate) |
| **codegen** | 9902 | Code generation (template, local, OpenAI) |
| **videogen** | 9903 | Video generation (AnimateDiff, built-in GIF) |
| **threed** | 9905 | 3D model generation (Shap-E, built-in OBJ) |
| **vision** | 9906 | Screen capture and image analysis |
| **voice** | 9907 | Unified speech-to-text + audio generation (pyttsx3, ElevenLabs, system) |
| **audiogen** | — | Audio/music generation |
| **transcriber** | — | Audio transcription |

## Quick Start

### Run a service standalone
```bash
# Generate an image directly
python mods/imagegen/imagegen.py --generate "a sunset over mountains"

# Start image service on port 9901
python mods/imagegen/imagegen.py --port 9901

# Speak text
python mods/voice/voice.py --speak "Hello world"

# Capture screen
python mods/vision/vision.py --capture
```

### Run with router (coordinated)
```bash
# Terminal 1: Start router
python mods/router/router.py

# Terminal 2: Start image service
python mods/imagegen/imagegen.py --router localhost:9900

# Terminal 3: Start code service
python mods/codegen/codegen.py --router localhost:9900

# Terminal 4: Check connected services
python mods/router/router.py --list
```

## Protocol

All services use TCP with JSON messages:

```python
# Message format
{
    "type": "command",  # register, command, response, event, heartbeat, shutdown
    "payload": {...},
    "id": "uuid"
}

# Length-prefixed: 4-byte big-endian length + JSON data
```

## Service Commands

### imagegen
- `generate` - Generate image from prompt
- `load_provider` - Load a provider (placeholder, local, openai, replicate)
- `unload_provider` - Unload a provider
- `list_providers` - List available providers
- `set_default` - Set default provider
- `status` - Get service status

### codegen  
- `generate` - Generate code from prompt
- `save_code` - Save generated code to file
- `load_provider` / `unload_provider` / `list_providers` / `set_default` / `status`

### videogen
- `generate` - Generate video from prompt
- `load_provider` / `unload_provider` / `list_providers` / `set_default` / `status`

### threed
- `generate` - Generate 3D model from prompt
- `load_provider` / `unload_provider` / `list_providers` / `set_default` / `status`

### vision
- `capture` - Capture screen
- `analyze` - Analyze image with AI
- `ocr` - Extract text from image
- `start_watch` / `stop_watch` - Auto-capture mode
- `load` / `status`

### voice
- `listen` - Listen for speech (microphone)
- `speak` - Text-to-speech
- `transcribe` - Transcribe audio file
- `generate_audio` - Generate audio file from text
- `start_continuous` / `stop_continuous` - Continuous listening mode
- `list_voices` / `set_voice` - Voice selection
- `set_rate` / `set_volume` - Speech settings
- `load_provider` / `unload_provider` / `list_providers` / `set_default` - Provider control
- `load` / `status`

## Providers

Each service supports multiple providers:

| Service | Providers |
|---------|-----------|
| imagegen | placeholder (built-in), local (Stable Diffusion), openai (DALL-E), replicate |
| codegen | template (built-in), local (Enigma), openai (GPT-4) |
| videogen | builtin (GIF), local (AnimateDiff), replicate |
| threed | builtin (OBJ primitives), local (Shap-E), replicate |
| vision | Built-in (mss/PIL/pyautogui for capture, tesseract/easyocr for OCR) |
| voice | speech (SpeechRecognition), whisper (local STT), pyttsx3/system/elevenlabs (TTS) |

## Environment Variables

```bash
# For cloud providers
OPENAI_API_KEY=sk-...
REPLICATE_API_TOKEN=r8_...
ELEVENLABS_API_KEY=...
```

## Python Client Example

```python
from mods.router.router import RouterClient

# Connect to router
client = RouterClient("localhost", 9900)
client.connect()

# List services
print(client.list_services())

# Generate image via router
result = client.send_command("imagegen", "generate", {
    "prompt": "a futuristic city",
    "width": 512,
    "height": 512
})
print(result)

# Or route by capability
result = client.route_by_capability("generate_image", "generate", {
    "prompt": "a cat"
})

client.disconnect()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         ROUTER                               │
│                       (port 9900)                            │
│                                                              │
│  Capabilities Registry    Service Registry                  │
│  generate_image -> [imagegen]    imagegen -> socket         │
│  generate_code -> [codegen]      codegen -> socket          │
│  tts -> [voice]                  ...                        │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┼──────────┬──────────┬──────────┬─────────┐
    │          │          │          │          │         │
┌───┴──┐  ┌───┴──┐  ┌───┴──┐  ┌───┴──┐  ┌───┴──┐  ┌──┴───┐
│image │  │code  │  │video │  │audio │  │3d    │  │vision│
│gen   │  │gen   │  │gen   │  │voice │  │3d    │  │vision│
│:9901 │  │:9902 │  │:9903 │  │:9907 │  │:9905 │  │:9906 │
└──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘
```

Each service:
1. Connects to router on startup (if --router specified)
2. Registers name, capabilities, and available commands
3. Receives commands via TCP from router
4. Sends responses back
5. Can also run standalone without router

## Output Directories

All services output to `outputs/` subdirectories:
- `outputs/images/` - Generated images
- `outputs/code/` - Generated code files  
- `outputs/videos/` - Generated videos/GIFs
- `outputs/voice/` - Generated audio files
- `outputs/3d/` - Generated 3D models
- `outputs/vision/` - Screen captures
- `outputs/voice/` - Voice recordings
