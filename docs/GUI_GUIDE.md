# 📖 Enigma Engine - Complete GUI Guide

**Everything you need to know to use the Enigma GUI, organized for easy reading and skimming.**

---

## Table of Contents

1. [🚀 Quick Start](#-quick-start)
2. [💬 Chat Tab](#-chat-tab---talk-to-your-ai)
3. [🎓 Train Tab](#-train-tab---teach-your-ai)
4. [📏 Scale Tab](#-scale-tab---choose-model-size)
5. [🎛️ Modules Tab](#%EF%B8%8F-modules-tab---control-ai-capabilities)
6. [🔀 Router Tab](#-router-tab---assign-models-to-tools)
7. [✨ Generation Tabs](#-generation-tabs---ai-generation-features)
8. [🎭 Avatar Tab](#-avatar-tab---visual-representation)
9. [👁️ Vision Tab](#%EF%B8%8F-vision-tab---see-your-screen)
10. [📷 Camera Tab](#-camera-tab---live-video-feed)
11. [💻 Terminal Tab](#-terminal-tab---command-line-access)
12. [📜 History Tab](#-history-tab---past-conversations)
13. [📁 Files Tab](#-files-tab---manage-training-data)
14. [⚙️ Settings Tab](#%EF%B8%8F-settings-tab---configure-resources)
15. [📋 Options Menu](#-options-menu)
16. [🔧 Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### First Launch

1. **Start the GUI**:
   ```bash
   python run.py --gui
   ```

2. **Setup Wizard** (first time only):
   - Enter a name for your AI (e.g., "Artemis", "Helper", "Bob")
   - Choose model size:
     - **Tiny** - Testing, Raspberry Pi
     - **Small** - Desktop, good balance (recommended)
     - **Medium** - Quality results, needs GPU
   - Click **Finish**

3. **Essential Modules Load Automatically**:
   - Model (AI brain)
   - Tokenizer (text processing)
   - Inference (response generation)

4. **Ready to Chat!**
   - Go to **Chat** tab
   - Type your message
   - Press Enter

---

## 💬 Chat Tab - Talk to Your AI

**What it does**: Main interface for conversing with your AI.

### Layout

```
┌────────────────────────────────────┐
│  [Chat History Display]            │
│  You: Hello!                       │
│  AI: Hello! How can I help?        │
│                                    │
└────────────────────────────────────┘
│ Type your message here...    [Send]│
└────────────────────────────────────┘
│ [🎤 Microphone] [🔊 Auto-Speak]    │
└────────────────────────────────────┘
```

### How to Use

1. **Type a message** in the input box at the bottom
2. **Press Enter** or click **Send**
3. **Wait for response** (shows in chat history)
4. **Continue the conversation** naturally

### Features

**📸 Upload Image**:
- Click the image icon
- Select an image file
- AI can analyze and discuss the image

**🎤 Voice Input** (if microphone enabled):
- Click microphone button
- Speak your message
- Auto-transcribed to text

**🔊 Auto-Speak** (if enabled):
- AI responses are spoken aloud
- Toggle in Options menu

**💾 Save Conversation**:
- Conversations auto-save to History
- Can load previous chats from History tab

### Tips

✅ **DO:**
- Ask follow-up questions (AI remembers context)
- Use natural language
- Request tool use: "generate an image", "search the web"

❌ **DON'T:**
- Expect instant responses with large models
- Overwhelm with extremely long messages
- Use before training (will give gibberish)

---

## 🎓 Train Tab - Teach Your AI

**What it does**: Train your model on text data to improve responses.

### Layout

```
┌────────────────────────────────────┐
│ Model: sacrifice (small)           │
│ Training Data: [Select File ▼]     │
│ Epochs: [30  ▼]  LR: [0.0001]     │
│ [Start Training] [Stop Training]   │
├────────────────────────────────────┤
│ Progress: [████░░░░░░] 45%         │
│ Epoch: 14/30                       │
│ Loss: 2.34  Time: 00:15:23         │
│                                    │
│ Training Log:                      │
│ Epoch 14/30 - Loss: 2.34          │
│ Epoch 13/30 - Loss: 2.56          │
└────────────────────────────────────┘
```

### How to Train

**Basic Training**:
1. **Select training data file** (or use default)
2. **Set epochs** (30 is good start)
3. **Click "Start Training"**
4. **Wait and monitor progress**
5. **Loss should decrease** (lower = better)

**Advanced Settings**:
- **Learning Rate**: How fast it learns (lower = slower but safer)
- **Batch Size**: How much data per step (higher = faster but more memory)
- **Save Checkpoints**: Auto-saves every 100 epochs

### Understanding the Display

**Loss Graph**:
- Shows learning progress
- Should go down over time
- Flattening = done learning

**Progress Bar**:
- Shows current epoch / total epochs
- Estimated time remaining

**Training Log**:
- Real-time updates
- Shows loss per epoch
- Any errors/warnings

### When to Stop Training

✅ **Stop when:**
- Loss drops below 1.5-2.0
- Loss stops decreasing (plateaus)
- Quality is good enough for your needs

⚠️ **Warning signs:**
- Loss increases = something wrong
- Loss stays very high (>5) = bad data or wrong settings
- Out of memory = reduce batch size

### Training Tips

**First Training** (new model):
- Use 30-50 epochs
- Monitor every 10 epochs
- Test in Chat after training

**Fine-tuning** (existing model):
- Use 10-20 epochs
- Lower learning rate (0.00001)
- Combine old + new data

**Quick Test**:
- Use 5 epochs with tiny model
- Verify training works
- Then do real training

---

## 📏 Scale Tab - Choose Model Size

**What it does**: Visualize and select model sizes from nano to omega.

### Visual Scale

Shows all 15 model sizes on a spectrum:
- **Embedded** (nano, micro) - Red
- **Edge** (tiny, mini) - Orange
- **Consumer** (small, medium, base) - Yellow
- **Prosumer** (large, xl) - Green
- **Server** (xxl, huge) - Blue
- **Datacenter** (giant, colossal) - Purple
- **Ultimate** (titan, omega) - Teal

### Model Cards

Each model shows:
- **Name** and **parameters** (e.g., SMALL ~27M)
- **Hardware tier** (Consumer, Server, etc.)
- **Description** (what it's good for)
- **Specs**: dimensions, layers, sequence length
- **Select button**

### How to Use

1. **Browse model cards** - Scroll through options
2. **Read descriptions** - Find one that fits your needs
3. **Click "Select"** on desired model
4. **Review hardware requirements** at bottom
5. **Click "Apply Model Size"** to change

### Hardware Requirements Display

Shows for selected model:
- **Minimum RAM** needed
- **CPU requirements**
- **GPU/VRAM** if applicable
- **Best for** use case

### Model Selection Guide

**Choose based on your hardware:**

| You Have | Recommended Size |
|----------|------------------|
| Raspberry Pi | tiny, mini |
| Laptop (no GPU) | small |
| Desktop (no GPU) | small, medium |
| GPU 4-8GB VRAM | medium, base, large |
| GPU 8-16GB VRAM | large, xl |
| GPU 16GB+ | xl, xxl |
| Multiple GPUs | xxl, huge, giant |
| Datacenter | colossal, titan, omega |

**Choose based on your goal:**

| Goal | Recommended Size |
|------|------------------|
| Learning/Testing | tiny, small |
| Personal chatbot | small, medium |
| Content creation | medium, large |
| Code assistant | large, xl |
| Production API | xl, xxl, huge |
| Research/Competition | giant, colossal, titan, omega |

### Growing Models

Can grow a model to next size:
1. Train small model first
2. Use Model Manager to grow
3. Knowledge transfers to larger size
4. Continue training

---

## 🎛️ Modules Tab - Control AI Capabilities

**What it does**: Enable/disable modules to add or remove AI features. Think of modules as "apps" for your AI.

### Layout

```
┌────────────────────────────────────┐
│ 🎛️ Module Manager                  │
│ Enable/disable any capability      │
│ [Search...] [↻ Refresh]            │
├────────────────────────────────────┤
│ ⚙️ CORE                             │
│ [All On] [All Off]                 │
│                                    │
│ ┌──────────┐ ┌──────────┐         │
│ │⚙️ Model  │ │📝 Token  │         │
│ │●On       │ │●On       │         │
│ │Core AI   │ │Text proc │         │
│ │[⚙]       │ │[⚙]       │         │
│ └──────────┘ └──────────┘         │
│                                    │
│ ✨ AI GENERATION                    │
│ [All On] [All Off]                 │
│                                    │
│ ┌──────────┐ ┌──────────┐         │
│ │🖼️ Image  │ │💻 Code   │         │
│ │○Off      │ │○Off      │         │
│ │Generate  │ │Generate  │         │
│ │images    │ │code      │         │
│ │[⚙]       │ │[⚙]       │         │
│ └──────────┘ └──────────┘         │
├────────────────────────────────────┤
│ Status:                            │
│ Loaded: 3 / 25                     │
│ CPU: 15% | Memory: 2.3GB           │
│                                    │
│ Activity:                          │
│ [15:23] Loading model...           │
│ [15:23] ✓ model enabled            │
└────────────────────────────────────┘
```

### Module Categories

**⚙️ CORE** (Essential - Auto-loaded):
- **Model** - AI brain, text generation
- **Tokenizer** - Converts text to numbers
- **Training** - Model training capabilities
- **Inference** - Response generation engine

**✨ AI GENERATION** (Add capabilities):
- **Image Gen (Local)** - Stable Diffusion on your GPU
- **Image Gen (Cloud)** - DALL-E, Replicate
- **Code Gen (Local)** - Code using your model
- **Code Gen (Cloud)** - GPT-4 code generation
- **Video Gen** - Create animations/videos
- **Audio/TTS** - Text-to-speech, music

**🧠 MEMORY**:
- **Memory Manager** - Save/recall conversations
- **Embeddings (Local)** - Semantic search
- **Embeddings (Cloud)** - OpenAI embeddings

**👁️ PERCEPTION**:
- **Voice Input** - Speech recognition
- **Vision** - Camera, image analysis

**📤 OUTPUT**:
- **Voice Output** - Text-to-speech
- **Avatar** - Visual AI representation

**🔧 TOOLS**:
- **Web Tools** - Search, fetch webpages
- **File Tools** - Read/write files

**🌐 NETWORK**:
- **API Server** - REST API endpoint
- **Multi-Device** - Distributed across devices

### How to Use Modules

**Enable a Module**:
1. Find the module card
2. **Click the toggle switch** (right side)
3. Wait for "Loading..." in activity log
4. Module turns **● On** when ready

**Disable a Module**:
1. Find an enabled module
2. **Click the toggle switch** to turn off
3. Module shows **○ Off**
4. Resources freed immediately

**Auto-Enable on Startup**:
- Core modules (Model, Tokenizer, Inference) load automatically
- Other modules start disabled
- Configure which ones to auto-load

### Module Details

**Click ⚙ button** on any card to:
- View module configuration
- See dependencies
- Check requirements
- Adjust settings

### Understanding Module States

**● On** (Green):
- Module is loaded and ready
- Using system resources
- Capabilities available

**○ Off** (Gray):
- Module not loaded
- Not using resources
- Capabilities unavailable

**⚠ Warning** (Yellow):
- Module loaded but issue detected
- Check activity log for details

**✗ Error** (Red):
- Module failed to load
- Missing dependencies or resources
- See activity log for reason

### Resource Monitoring

**Status Panel** (right side) shows:

**Loaded Modules**:
- "Loaded: 3 / 25" = 3 active out of 25 available

**System Resources**:
- **CPU**: Processor usage %
- **Memory**: RAM used
- **GPU**: Graphics card usage (if available)
- **VRAM**: Graphics memory used

**Activity Log**:
- Real-time module loading/unloading
- Success/failure messages
- Error details

### Module Tips

**When to Enable**:
- ✅ Enable before using a feature
- ✅ AI will ask to enable if needed
- ✅ Load heavy modules when needed, unload after

**When to Disable**:
- ✅ When not using a feature
- ✅ If running low on memory
- ✅ To free up GPU for gaming/other apps
- ✅ If computer is slowing down

**Resource Management**:
- Image/Video generation = Heavy (2-4GB VRAM)
- Code generation = Light (<500MB)
- Voice = Light (<200MB)
- Memory = Light (<100MB)

**Conflicts to Avoid**:
- Don't load both Local AND Cloud versions of same feature
  - Either `image_gen_local` OR `image_gen_api`, not both
  - They provide same capability, will conflict

### Common Module Combinations

**Basic Chatbot** (minimal):
```
✓ model
✓ tokenizer  
✓ inference
```

**Smart Assistant** (recommended):
```
✓ model
✓ tokenizer
✓ inference
✓ memory
✓ web_tools
✓ file_tools
```

**Creative AI**:
```
✓ model
✓ tokenizer
✓ inference
✓ image_gen_local (or _api)
✓ code_gen_local
✓ memory
```

**Full Featured**:
```
✓ model
✓ tokenizer
✓ inference
✓ memory
✓ embedding_local
✓ image_gen_local
✓ code_gen_local
✓ voice_input
✓ voice_output
✓ avatar
✓ vision
✓ web_tools
✓ file_tools
```

### AI Self-Management

Your AI can now control modules itself! In chat, you can say:
- "Check if you have enough memory"
- "Turn on image generation"
- "What modules do you have?"
- "Disable video generation to free memory"

The AI will use the `load_module`, `unload_module`, `list_modules`, and `check_resources` tools automatically.

---

## 🔀 Router Tab - Assign Models to Tools

**What it does**: Control which AI model handles each capability (chat, image, code, etc.).

### Why Use Router?

Different models are good at different things:
- **DialoGPT** → Great for casual conversation
- **Qwen/Qwen2** → Better for following instructions
- **Codegen** → Specialized for code
- **Your trained Enigma model** → Custom personality

The Router lets you assign the best model for each task!

### Layout

**Quick Assign** (top):
- Select a model from dropdown
- Click **⚡ Apply to ALL Tools** to use it everywhere
- Great for quickly switching your entire AI

**Tool Cards** (grid):
- Each tool (Chat, Image, Code, etc.) has its own card
- Shows currently assigned model(s)
- Set priority for fallback order

### How to Use

**Assign One Model to All Tools**:
1. In **Quick Assign** section, select a model
2. Click **⚡ Apply to ALL Tools**
3. Confirm when prompted
4. All tools now use that model

**Assign Different Models to Different Tools**:
1. Find the tool card (e.g., CHAT, CODE)
2. Select a model from its dropdown
3. Click **+** to add it
4. Click **Save Configuration**

**Add HuggingFace Model**:
1. In **Add HuggingFace Model** section
2. Enter model ID (e.g., `microsoft/DialoGPT-medium`)
3. Click **Add to List**
4. Model appears in all dropdowns

### Model Sources

Models can come from:
- **Local Enigma** - Your trained models (e.g., `pi_tiny`)
- **HuggingFace** - Prefixed with `huggingface:` (e.g., `huggingface:Qwen/Qwen2-0.5B-Instruct`)
- **API** - Cloud services like OpenAI

### Priority & Fallback

If you assign multiple models to one tool:
- Higher priority number = tried first
- If model fails/unavailable, next one is tried
- Example: Priority 100 → Priority 50 → Priority 10

### Tips

- **Raspberry Pi**: Use `Qwen/Qwen2-0.5B-Instruct` (503M) - small but capable
- **Desktop**: Try `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for better quality
- **GPU Available**: Can use larger 7B+ models
- Click **Reset to Defaults** if things go wrong

---

## ✨ Generation Tabs - AI Generation Features

**What it does**: Test and configure AI generation capabilities (images, code, video, audio, 3D, embeddings).

### Generation Tabs

**🖼️ Image Tab**:
- Generate images from text prompts
- Choose local (Stable Diffusion) or cloud (DALL-E, Replicate)
- Set width/height
- Save generated images

**💻 Code Tab**:
- Generate code from descriptions
- Choose programming language
- Use local model or cloud (GPT-4)
- Copy or save code

**🎬 Video Tab**:
- Generate short video clips
- Set duration
- Local (AnimateDiff) or cloud (Replicate)
- Save as MP4 or GIF

**🎵 Audio Tab**:
- Text-to-speech
- Local (pyttsx3) or cloud (ElevenLabs, Replicate)
- Set voice, rate, volume
- Save as audio file

**🔢 Embeddings Tab**:
- Generate vector embeddings
- Compare text similarity
- Local (sentence-transformers) or cloud (OpenAI)
- Export embeddings to JSON

**🎲 3D Tab**:
- Generate 3D models from text
- Local (Shap-E) or cloud (Replicate)
- Save as PLY/GLB/OBJ

### How to Use

1. **Select tab** for feature you want
2. **Choose provider** (local vs cloud)
   - Local = runs on your machine, private, unlimited
   - Cloud = high quality, costs money, needs API key
3. **Enter prompt** describing what you want
4. **Adjust settings** (size, duration, etc.)
5. **Click "Generate"**
6. **Wait for result**
7. **Save** if you like it

### Provider Setup

**Local Providers**:
- No API key needed
- Uses your GPU
- Free, unlimited
- Enable corresponding module first

**Cloud Providers**:
- Need API key (from OpenAI, Replicate, etc.)
- Click **Configure** to enter key
- Costs per generation
- Higher quality

---

## 🎭 Avatar Tab - Visual Representation

**What it does**: Shows a visual representation of your AI.

### Features

- **Enable/Disable**: Toggle avatar on/off
- **Expressions**: Changes based on AI mood
- **Speech Animation**: Moves when talking (if voice enabled)
- **Customization**: Change appearance (future feature)

### How to Use

1. **Enable in Options menu**: Options → Avatar
2. **Avatar appears** in tab
3. **Expressions update** during conversation
4. **Speaks** when voice output enabled

---

## 👁️ Vision Tab - See Your Screen

**What it does**: Allows AI to capture and analyze what's on your screen or from camera.

### Features

**📸 Screenshot**:
- Capture current screen
- AI can analyze and describe
- Use in chat: "What's on my screen?"

**📷 Camera**:
- Capture from webcam
- Real-time or snapshot
- AI can identify objects, read text

**🖼️ Upload Image**:
- Load image file
- AI analyzes and discusses
- Useful for identifying things

### How to Use

1. **Capture Method**:
   - Click **Screenshot** for screen capture
   - Click **Camera** for webcam
   - Click **Load Image** for file

2. **Image Displays** in preview area

3. **AI Analysis**:
   - Automatic OCR (text extraction)
   - Click **Analyze** for AI description
   - Or go to Chat and ask about it

### Use Cases

- "What's in this image?"
- "Read the text on screen"
- "Identify this object"
- "What game am I playing?"
- "Describe this picture"

---

## 📷 Camera Tab - Live Video Feed

**What it does**: Live camera preview with photo/video capture and AI analysis.

### Features

**🎥 Live Preview**:
- ~30 FPS real-time camera feed
- Multiple camera support (0, 1, 2)
- Resolution options (320x240 to 1280x720)

**📸 Capture Photo**:
- Save current frame as JPG
- Saved to `information/images/` folder
- Timestamped filenames

**⏺ Record Video**:
- Record to AVI file
- Toggle on/off recording
- Visual indicator when recording

**🤖 AI Analysis**:
- Have AI describe what it sees
- Auto-analyze on interval
- Works with vision tools

### How to Use

**Start Camera**:
1. Select camera (0 = default, 1, 2 for additional)
2. Choose resolution
3. Click **▶ Start Camera**
4. Live feed appears in preview

**Capture Photo**:
1. With camera running, click **📸 Capture Photo**
2. Image saved with timestamp
3. Status shows save location

**Record Video**:
1. With camera running, click **⏺ Record Video**
2. Button changes to **⏹ Stop Recording**
3. Click again to stop
4. Video saved as AVI

**AI Analysis**:
1. Click **🤖 AI Analyze** for one-time analysis
2. Or enable **Auto-analyze every X sec**
3. Results appear in Analysis section

### Camera vs Vision Tab

| Feature | Vision Tab | Camera Tab |
|---------|-----------|------------|
| Screen capture | ✅ Yes | ❌ No |
| Single camera shot | ✅ Yes | ✅ Yes |
| Live preview | ❌ No | ✅ Yes |
| Video recording | ❌ No | ✅ Yes |
| Auto-watch | ✅ Screen | ✅ Camera |

### Requirements

- **OpenCV** must be installed:
  ```bash
  pip install opencv-python
  ```
- Camera connected and accessible
- Raspberry Pi: Use official camera module or USB webcam

---

## 💻 Terminal Tab - Command-Line Access

**What it does**: Embedded terminal for running commands directly.

### Features

- Run Python scripts
- Test modules
- Debug issues
- System commands

### How to Use

1. Type command at prompt
2. Press Enter
3. Output shows in terminal
4. Like a normal command line

**Useful Commands**:
```bash
python -m pytest tests/ -v      # Run test suite
python run.py --train           # Train from terminal
pip list                        # Show installed packages
```

---

## 📜 History Tab - Past Conversations

**What it does**: View and load previous chat sessions.

### Layout

```
┌────────────────────────────────────┐
│ Conversation History               │
├────────────────────────────────────┤
│ ▶ 2026-01-02 15:23 (15 messages)  │
│ ▶ 2026-01-02 10:10 (23 messages)  │
│ ▶ 2026-01-01 18:45 (8 messages)   │
└────────────────────────────────────┘
│ [Load] [Delete] [Export]           │
└────────────────────────────────────┘
```

### How to Use

1. **View list** of past conversations
2. **Click conversation** to see preview
3. **Load**: Restores chat in Chat tab
4. **Delete**: Removes conversation
5. **Export**: Save as text file

### Features

- Auto-saves all conversations
- Timestamp and message count
- Search conversations
- Organize by date

---

## 📁 Files Tab - Manage Training Data

**What it does**: View and edit training data files.

### Features

**File List**:
- Shows all `.txt` files in `data/` directory
- Click to view/edit
- Create new files

**Editor**:
- Syntax highlighting for Q&A format
- Line numbers
- Save changes
- Real-time preview

### How to Use

**Edit Existing File**:
1. Select file from list
2. Edit in text editor
3. Click **Save**
4. Retrain model to apply changes

**Create New File**:
1. Click **New File**
2. Enter filename
3. Add training data
4. Save
5. Use in Train tab

### Training Data Format

```
Q: Question here?
A: Answer here.

Q: Another question?
A: Another answer.
```

---

## ⚙️ Settings Tab - Configure Resources

**What it does**: Control how much CPU/GPU the AI uses, manage power consumption.

### Sections

**🖥️ Hardware Detection**:
- Shows detected GPU
- CPU thread count
- Available memory

**⚡ Power Mode**:
- **Minimal** - Best for gaming (AI uses <10% resources)
- **Gaming** - AI in background (AI uses ~25%)
- **Balanced** - Normal use (AI uses ~50%)
- **Performance** - Faster AI (AI uses ~75%)
- **Maximum** - Use all resources (AI uses ~100%)

**🤖 Autonomous Mode**:
- Allow AI to act independently
- Learn from web
- Explore curiosities
- Evolve personality

### How to Use

1. **Select Power Mode** from dropdown
2. **Changes apply immediately**
3. **Monitor resource usage**
4. **Adjust based on needs**

### When to Adjust

**Lower power mode** when:
- Gaming or doing heavy tasks
- Computer running hot
- Battery life important
- AI responses can be slower

**Higher power mode** when:
- Want faster AI responses
- Computer idle otherwise
- Training models
- Not concerned about power

---

## 📋 Options Menu

Located at top menu bar: **Options**

### Theme Submenu

Choose visual theme:
- **Dark (Catppuccin)** - Default, easy on eyes
- **Light** - Bright, high contrast
- **Shadow (Deep Purple)** - Dark purple
- **Midnight (Deep Blue)** - Navy blue

### Toggle Options

**Avatar (OFF/ON)**:
- Enable/disable avatar display
- Shows in Avatar tab and during chats

**AI Auto-Speak (OFF/ON)**:
- AI speaks responses aloud
- Requires voice output module
- Uses text-to-speech

**Microphone (OFF/ON)**:
- Enable voice input
- Speak instead of typing
- Requires voice input module

**Learn While Chatting (ON/OFF)**:
- AI learns from conversations
- Updates model gradually
- Can be disabled for consistency

### How to Use

1. Click **Options** in menu bar
2. Click option to toggle
3. State shows in name (OFF/ON)
4. Changes apply immediately

---

## 🔧 Troubleshooting

### Scale Tab is Hard to Read

**Problem**: Text too small, low contrast

**Solution**: Already fixed in latest version!
- Increased font sizes (11-13px)
- Better color contrast (#cdd6f4)
- All text is now selectable/copyable

### Module Won't Load

**Problem**: Module stays "Loading..." or shows error

**Solutions**:
1. Check Activity Log for specific error
2. Install missing dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Check if conflicting module loaded (e.g., both local and cloud image gen)
4. Verify hardware requirements met (GPU for some modules)
5. Restart GUI

### Out of Memory

**Problem**: "CUDA out of memory" or system freeze

**Solutions**:
1. **Unload heavy modules**:
   - image_gen_local
   - video_gen_local
   - embedding_local
2. **Use smaller model size**
3. **Close other GPU applications**
4. **Lower power mode** in Settings
5. **Restart GUI** to clear memory

### AI Gives Gibberish

**Problem**: Responses make no sense

**Solutions**:
1. **Check if model is trained**
   - New models need training first
   - See Train tab
2. **Train for more epochs** (30-50 minimum)
3. **Verify training data quality**
4. **Check model size** (tiny models limited)

### Training Fails

**Problem**: Training stops with error

**Solutions**:
1. **Check training log** for specific error
2. **Verify data file format** (Q:/A: pairs)
3. **Ensure sufficient disk space** (10GB+ free)
4. **Lower batch size** if out of memory
5. **Close other applications**

### Modules Not Working in Chat

**Problem**: AI says it can't use tools

**Solutions**:
1. **Enable the module first** (Modules tab)
2. **Verify module loaded** (shows ● On)
3. **Check if tools enabled** in inference settings
4. **Retrain with tool usage examples**

### GUI Won't Start

**Problem**: `python run.py --gui` fails

**Solutions**:
1. **Install PyQt5**:
   ```bash
   pip install PyQt5
   ```
   On Raspberry Pi:
   ```bash
   sudo apt install python3-pyqt5
   ```
2. **Check Python version** (3.8+ required)
3. **Verify all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Reference

### Common Tasks

| Task | Where | How |
|------|-------|-----|
| Chat with AI | Chat tab | Type message, press Enter |
| Train model | Train tab | Select data, set epochs, click Start |
| Change model size | Scale tab | Select model, click Apply |
| Enable feature | Modules tab | Find module, toggle switch on |
| Generate image | Image tab | Enter prompt, click Generate |
| Generate code | Code tab | Describe code, click Generate |
| Text-to-speech | Audio tab | Enter text, click Speak/Save |
| View past chats | History tab | Click conversation to load |
| Edit training data | Files tab | Select file, edit, save |
| Adjust resources | Settings tab | Choose power mode |
| Change theme | Options menu | Theme → Select |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message (Chat tab) |
| `Shift+Enter` | New line (Chat tab) |
| `Ctrl+S` | Save (Files tab) |
| `F5` | Refresh modules |

### Status Indicators

| Symbol | Meaning |
|--------|---------|
| ● Green | Active/Loaded |
| ○ Gray | Inactive/Unloaded |
| ⚠ Yellow | Warning |
| ✗ Red | Error |
| ✓ Green | Success |
| ↻ | Loading/Refreshing |

---

## Related Documentation

- **[Getting Started](GETTING_STARTED.md)** - Quick 5-minute setup
- **[How to Train](HOW_TO_TRAIN.md)** - Detailed training guide
- **[Module Guide](MODULE_GUIDE.md)** - Module system details
- **[Tool Use](TOOL_USE.md)** - Teaching AI to use tools

---

**Need help?** Check troubleshooting section or create an issue on GitHub.

**Happy AI building!** 🚀🤖
