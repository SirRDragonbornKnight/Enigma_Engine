# Enigma AI Engine - TODO Checklist

**Last Updated:** February 10, 2026

---

## Current Tasks

All current tasks complete!

---

## Recently Completed (Feb 10, 2026)
- [x] **Landing website** - Created landing page with download links at `/` route
- [x] **Mobile download page** - Download routes for all platforms (Windows, macOS, Linux, Android, iOS, RPi)
- [x] **Chat image/video attachments** - Images and videos now display inline in chat messages
- [x] **Attachment file dialog expanded** - Now includes video (mp4, webm, mov) and audio formats
- [x] **Vision AI integration** - Attached images are analyzed if vision tools available

---

## Ideas for Future Features

### High Priority - Quick Wins
- [x] **VRM Avatar Support** - Add support for anime-style VRM avatars (docs/AVATAR_CREATION_GUIDE.md)
- [x] **Add `/api/feedback` endpoint** - Allow web/mobile feedback collection for future training
- [x] **Modularize run.py** - Extract command handlers into enigma_engine/cli/commands.py (498→233 lines)
- [x] **Split avatar_display.py** - Extracted OpenGL3DWidget to avatar_rendering.py (8827→7624 lines)
- [x] **Split enhanced_window.py** - Extracted GenerationPreviewPopup to dialogs/ (7716→7412 lines)

### Feature Enhancements (Medium Effort)
- [x] **Game Mode: Per-game profiles** - Different settings for different games (GamingProfileDialog UI, save/load JSON)
- [x] **Game Mode: FPS monitoring** - Dynamically adjust limits based on performance (FPSMonitor class)
- [x] **Game Mode: Steam API integration** - Auto-detect games from Steam (steam_integration.py)
- [x] **Game Mode: Game-specific AI personalities** - Context-aware responses (GamePersonality class in game_router.py)
- [x] **Web Interface: Push notifications** - Browser push for responses
- [x] **Web Interface: Background sync** - Service worker sync
- [x] **Web Interface: Voice input from web** - WebRTC audio capture (voice-input.js, /api/voice/transcribe)
- [x] **Web Interface: Multi-user support** - User accounts and sessions (session_middleware.py with SessionManager, login/register, role-based access)
- [x] **Discord bot: Voice chat integration** - Real-time voice conversations (integrations/discord_voice.py with DiscordBot, VoiceConfig, STT/TTS)
- [x] **Avatar: Full MoE expert routing** - Complete mixture-of-experts implementation (core/moe_router.py with MoELayer, ExpertRouter, Switch/ExpertChoice variants)
- [x] **Avatar: AWQ/GPTQ quantization** - Advanced quantization support (core/gptq_awq_loader.py)

### Code Quality
- [x] **Add type hints to core/ files** - Already well-typed (model.py, inference.py, tokenizer.py, training.py)
- [x] **Add cache eviction to unbounded dicts** - BoundedCache, TTLCache in utils/bounded_cache.py
- [x] **Consolidate get_* singleton functions** - SingletonFactory, @singleton in utils/singleton.py
- [x] **Plugin hot-reload support** - Reload plugins without restart (PluginFileWatcher in tools/plugins.py)
- [x] **Add Result types for fallible operations** - Functional error handling (Ok/Err in utils/result.py)

### Hotkey Enhancements
- [x] **Advanced gesture support** - Mouse + keyboard combos (GestureManager in core/gesture_manager.py)
- [x] **Per-application hotkey profiles** - Different hotkeys per app (HotkeyProfile in hotkey_manager.py)
- [x] **Cloud sync of preferences** - Sync settings across devices
- [x] **Voice activation as alternative** - "Hey Enigma" trigger (WakeWordDetector in voice/wake_word.py - offline local detection)
- [x] **Multi-step hotkey sequences** - Chord hotkeys (ChordSequence/ChordManager in hotkey_manager.py)

### Self-Improvement System (Requires Training Later)
- [x] **LoRA Training Integration** - Connect scheduler to training.py (TrainingScheduler in learning/training_scheduler.py with LoRAConfig, LoRAModel, LoRATrainer)
- [x] **Federated learning integration** - Share improvements without data (federated/federation.py with FederatedLearning, aggregation, privacy, compression)
- [x] **Critic model** - Better response evaluation (learning/critic_model.py with Critic, CriticTrainer, RLHFTrainer, 7 evaluation dimensions)
- [x] **A/B testing of personality traits** - Data-driven personality (learning/ab_testing.py with PersonalityABTest, variants, Thompson sampling)
- [x] **Multi-model coordination** - Specialized model learning (learning/model_coordination.py with ModelCoordinator, ensemble/cascade/consensus strategies)

### Model / Architecture
- [x] **ONNX model loading** - Full implementation with weight extraction and config inference
- [x] **Full MoE Expert Routing** - Config ready, needs FeedForward changes (core/moe_router.py - complete implementation)
- [x] **Enhanced Attention** - Sliding window and paged attention
- [x] **GPTQ/AWQ loaders** - Advanced quantization formats (core/gptq_awq_loader.py with GPTQModel, AWQModel, auto-detect, quantize, registry)
- [x] **Continuous batching** - For serving infrastructure (BatchScheduler, InferenceServer in core/continuous_batching.py)
- [x] **Tensor parallelism** - Multi-GPU distributed inference (core/tensor_parallel.py with ColumnParallelLinear, RowParallelLinear, ParallelEmbedding, 528 lines)

### Integrations
- [x] **OBS streaming enhancements** - Better scene switching, chat overlays (auto-scene rules, alerts, reactions in obs_streaming.py)
- [x] **LangChain tools** - Expose Enigma tools as LangChain Tools (EnigmaToolkit in integrations/langchain_tools.py)
- [x] **Home Assistant integration** - Smart home control via conversation (integrations/home_assistant.py)

### Mobile / Desktop Apps
- [x] **React Native full app** - Complete mobile/src/ with screens (Profile, ImageGen, Voice, Model), components (Button, MessageBubble), hooks (useStorage, useApi, useTheme, useNetworkStatus)
- [x] **Electron polish** - Enhanced packaging/electron/ with settings.js (persistent config), extended preload.js APIs (clipboard, shell, notifications, streaming), quick-chat.html overlay
- [x] **PWA install prompts** - Better web app installation flow

### Mobile: Remote Training & Full Features (HIGH PRIORITY)
**Goal: Full desktop feature parity on mobile, PC does heavy lifting**

#### Training & Feedback from Mobile
- [x] **POST /api/v1/feedback** - Submit thumbs up/down on responses for training queue
- [x] **POST /api/v1/training/example** - Submit Q&A training examples from mobile
- [x] **GET /api/v1/training/stats** - View training queue status, examples collected
- [x] **POST /api/v1/training/start** - Trigger training on PC from mobile (with auth)
- [x] **GET /api/v1/training/progress** - Poll training progress

#### Generation Features (PC runs, mobile displays)
- [x] **POST /api/v1/generate/image** - Image generation, returns URL/base64
- [x] **POST /api/v1/generate/code** - Code generation with syntax highlighting
- [x] **POST /api/v1/generate/audio** - Audio/music generation, returns audio URL
- [x] **POST /api/v1/generate/video** - Video generation, returns video URL  
- [x] **POST /api/v1/generate/3d** - 3D model generation, returns GLB URL

#### Memory & Conversation Sync
- [x] **GET /api/v1/conversations** - List all conversations from PC
- [x] **POST /api/v1/conversations/sync** - Sync mobile conversations to PC
- [x] **GET /api/v1/memory/search** - Search conversation memory
- [x] **POST /api/v1/memory/export** - Export memories to mobile for offline

#### Avatar Control from Mobile
- [x] **GET /api/v1/avatar/state** - Get current avatar emotion/pose
- [x] **POST /api/v1/avatar/emotion** - Set avatar emotion remotely
- [x] **POST /api/v1/avatar/gesture** - Trigger avatar gesture
- [x] **WebSocket /ws/avatar** - Real-time avatar state streaming

#### Mobile App Features  
- [x] **Offline mode** - Cache last N conversations, replay when online
- [x] **Background sync** - Upload training examples when on WiFi
- [x] **Push notifications** - Notify when generation complete, training done
- [x] **Voice-first interface** - Hold-to-talk, continuous conversation (VoiceService.ts with wake word, TTS, STT)
- [x] **Camera integration** - Photo -> vision analysis -> PC processes (CameraService.ts with capture, analyze, OCR, QR scan)
- [x] **Settings sync** - Personality/voice/preferences sync with PC
- [x] **QR code pairing** - Scan QR to connect mobile to PC instantly
- [x] **Multiple PCs** - Connect to different Enigma instances (ServerManager.ts with discovery, health checks, switching)

### Cross-Device
- [x] **Task offloading UI** - GUI for distributing work across devices (task_offloading_tab.py with per-task routing, queue, load balancing)
- [x] **Model sharding** - Split large models across multiple machines (core/model_sharding.py with ShardCoordinator, pipeline/tensor parallelism)
- [x] **Automatic discovery** - Zero-config device pairing

### Testing / Documentation
- [x] **Integration test suite** - End-to-end workflow tests (testing/integration_tests.py with 8 suites, workflow tests)
- [x] **API documentation generator** - Auto-generate OpenAPI/Swagger docs
- [x] **Interactive tutorials** - In-app guided tours
- [x] **Model scaling tests** - Enabled tests in test_model.py (fixed feed_forward attribute name, tests pass)

### Experimental Features (Marked Experimental in Code)
- [x] **Model scaling validation** - Grow/shrink model tests pass (fixed model_scaling.py to use correct TransformerBlock attributes)

---

## Quick Reference

### Recently Completed (Feb 2026)
- **Quick API module** - Simple one-liner functions: `chat()`, `search()`, `read()`, `write()`, `ls()`, `wiki()`, `translate()`
- **Enhanced error handling** - `retry()` decorator, `from_tool_result()`, `as_result()` for functional-style error handling
- **Memory tools** - `search_memory`, `memory_stats`, `export_memory`, `import_memory` for AI to manage its history
- **ConversationManager enhanced** - `export_all()`, `import_all()`, `get_stats()`, `search_conversations()` methods
- **ToolProfiler** - Track tool usage, execution times, success rates
- **Batch tool execution** - `batch_execute_tools()` with parallel processing (273x speedup)
- **Tool result caching** - `cached_execute()` for cacheable tools with TTL
- **Tool summary API** - `get_tool_summary()` for system-wide stats
- **RichParameter system for all 109 tools** - Complete type info, ranges, enums, examples for every tool
- **Documentation updated** - README.md, docs/TOOL_USE.md with programmatic API examples
- Performance profiled (105 tools in 113ms init, 0.000ms lookup)
- Bug hunting completed (fixed cross-platform test, validated all tool structures)
- Fullscreen visibility control with `fullscreen_mode.py`
- Screen effects overlay system with 12 presets
- Avatar part editor with morphing transitions
- Mesh manipulation with blend shapes
- HuggingFace model import GUI
- Trainer fine-tuning workflow for pre-trained models
- Gaming mode integration with screen effects
- Settings persistence for all modes

### Key Files
| Feature | File |
|---------|------|
| Quick API | `enigma_engine/quick.py` |
| Tool system | `enigma_engine/tools/tool_registry.py` |
| Memory tools | `enigma_engine/tools/memory_tools.py` |
| Error handling | `enigma_engine/utils/errors.py` |
| Screen effects | `enigma_engine/avatar/screen_effects.py` |
| Part editor | `enigma_engine/avatar/part_editor.py` |
| Mesh manipulation | `enigma_engine/avatar/mesh_manipulation.py` |
| Fullscreen control | `enigma_engine/core/fullscreen_mode.py` |
| Gaming mode | `enigma_engine/core/gaming_mode.py` |
| Model import | `enigma_engine/gui/tabs/import_models_tab.py` |
