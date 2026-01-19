# ForgeAI - How Files Connect 🔗

This shows how each file talks to other files.

---

## 🚀 The Startup Flow

When you run ForgeAI, here's what happens:

```
run.py
   │
   ├── --train  →  training.py  →  model.py + tokenizer.py
   │
   ├── --gui    →  enhanced_window.py  →  inference.py  →  model.py
   │
   └── --serve  →  api_server.py  →  inference.py  →  model.py
```

---

## 🧠 The Brain (`core/`)

How the AI files connect:

### `model.py` (the neural network)
```
Used by:
  ├── training.py (to train it)
  ├── inference.py (to generate text)
  └── model_registry.py (to manage multiple models)
```

### `tokenizer.py` (text to numbers)
```
Used by:
  ├── training.py
  └── inference.py
```

### `inference.py` (text generation)
```
Uses:
  ├── model.py
  ├── tokenizer.py
  └── tool_router.py

Used by:
  ├── chat_tab.py
  └── api_server.py
```

### `tool_router.py` (decides which tool to use)
```
Uses:
  └── tokenizer.py

Routes to:
  ├── image_tab.py
  ├── code_tab.py
  ├── video_tab.py
  └── etc.
```

---

## 🖥️ The GUI (`gui/`)

How the interface connects:

### `enhanced_window.py` (main window)
```
Contains:
  └── All tabs

Uses:
  └── inference.py (for AI responses)
```

### Tab Connections

| Tab | Uses |
|-----|------|
| `chat_tab.py` | inference.py, memory/manager.py, voice/ |
| `image_tab.py` | StableDiffusionLocal, OpenAIImage, builtin/ |
| `code_tab.py` | inference.py, OpenAICode |
| `avatar_tab.py` | avatar/controller.py, avatar/autonomous.py |
| `modules_tab.py` | modules/manager.py |

---

## 💾 Memory (`memory/`)

How conversations are saved:

```
manager.py (ConversationManager)
   │
   ├── Saves to: data/conversations/*.json
   │
   └── Uses: vector_db.py (for semantic search)
                  │
                  └── Uses: embeddings.py
```

---

## ⚙️ Modules (`modules/`)

How the module system works:

```
manager.py (ModuleManager)
   │
   ├── Reads: registry.py (list of all modules)
   │
   ├── Saves to: data/module_config.json
   │
   └── Used by: modules_tab.py (GUI toggle)
```

### What registry.py defines:

| Module Class | Wraps |
|-------------|-------|
| `ModelModule` | model.py |
| `TokenizerModule` | tokenizer.py |
| `ImageGenLocalModule` | image_tab.py |
| `CodeGenModule` | code_tab.py |
| ... | ... |

---

## 🎭 Avatar (`avatar/`)

How the avatar connects:

```
controller.py (AvatarController)
   │
   ├── Uses: animation_system.py (movement)
   │
   ├── Uses: lip_sync.py (mouth movement)
   │
   └── Used by: avatar_tab.py


autonomous.py (self-acting avatar)
   │
   ├── Uses: controller.py
   │
   ├── Watches: screen content
   │
   └── Changes: mood based on what it sees
```

---

## 🌐 Networking (`comms/`)

How networking works:

### API Server
```
api_server.py (REST API)
   │
   ├── Uses: inference.py
   │
   └── Exposes: /health, /generate endpoints
```

### Multi-device
```
network.py
   │
   ├── Uses: model.py (to share models)
   │
   └── Uses: discovery.py (to find other devices)
```

---

## 🔊 Voice (`voice/`)

How speech works:

### Text-to-Speech
```
voice_generator.py
   │
   ├── Uses: voice_profile.py (settings)
   │
   ├── Uses: personality.py (affects voice)
   │
   └── Used by: audio_tab.py
```

### Speech-to-Text
```
listener.py
   │
   └── Outputs to: chat_tab.py
```

---

## 🔄 Complete Data Flow

What happens when you send a message:

```
1. You type in chat_tab.py
         │
         ▼
2. Message goes to tool_router.py
         │
         ▼
3. Router decides: "Is this a tool request or chat?"
         │
         ├─── If CHAT ────────────────────┐
         │                                │
         │    4. tokenizer.py (text → numbers)
         │              │
         │    5. model.py (process)
         │              │
         │    6. inference.py (generate)
         │              │
         │    7. tokenizer.py (numbers → text)
         │              │
         │    8. Back to chat_tab.py
         │
         └─── If TOOL (like "draw a cat") ─┐
                                           │
              4. tool_executor.py
                        │
              5. Right tool (image_tab.py)
                        │
              6. Tool generates result
                        │
              7. Back to chat_tab.py
```

---

## 📌 Summary: Key Connections

| File | Role |
|------|------|
| `run.py` | Everything starts here |
| `model.py` | The AI brain, used by training and inference |
| `inference.py` | Generates text, used by GUI and API |
| `tokenizer.py` | Text/numbers, used everywhere |
| `tool_router.py` | Routes requests to the right place |
| `manager.py` | Loads/unloads modules |
| `registry.py` | Defines all available modules |
