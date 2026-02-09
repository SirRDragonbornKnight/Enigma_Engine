# Enigma AI Engine - TODO

**Last Updated:** February 8, 2026

---

## Current Goal: API-Powered Trainer AI System

Use an API key to train a Trainer AI that can teach any combination of tasks to local AIs.

### Core Vision
```
API (GPT-4/Claude) → Training Data → Local Trainer AI → Specialized AIs
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
              All-in-One AI        Combo AIs (chat+code)    Specialist AIs
```

### What's Done
- [x] `APITrainingProvider` class - generates training data from APIs
- [x] `TRAINING_TASKS` - 12 task types (chat, code, vision, avatar, image_gen, audio_gen, video_gen, 3d_gen, game, robot, math, router)
- [x] Secure API key storage - encrypted at rest in `data/.secrets/`
- [x] Auto-load keys from secure storage
- [x] TrainerAI integration methods
- [x] **Multiple API keys support** - Store as many keys as you want with custom names
- [x] **Key management GUI** - View, use, and delete keys easily

### What's Next
- [x] **GUI for API Training** - Added to Build AI tab (Step 5: Training Data)
  - **Stored Keys section** - View all saved keys, use selected, delete with confirmation
  - **Add New Key section** - Provider dropdown, key name field, save button
  - API provider dropdown (OpenAI/Anthropic/Custom)
  - API key input with show/hide toggle
  - Task selection checkboxes for all 12 tasks
  - Quick-select presets (All, Basic, Creative, None)
  - Examples per task spinbox (10-500)
  - Generate button with progress bar
  - Cancel button for long operations

- [x] **Test the full workflow** - Validated with dry-run test
  - `scripts/test_full_workflow.py` - Tests all components without API calls
  - Secure key storage ✓
  - API configuration ✓
  - Mock data generation ✓
  - Model training ✓
  - Model inference ✓
  - GUI components ✓

- [x] **Create base model for GitHub releases**
  - Script: `scripts/train_base_release_model.py`
  - Trains "small" (~27M params) model on curated dataset
  - Combines 7 data sources: base knowledge, self-awareness, personality, instructions, router, code, avatar
  - Saves model with full metadata for easy loading
  - Usage: `python scripts/train_base_release_model.py` (or `--size tiny --epochs 10` for testing)

---

## Quick Reference

### API Training Usage
```python
from enigma_engine.core.trainer_ai import get_trainer_ai

trainer = get_trainer_ai()

# First time: provide API key (stored securely)
trainer.configure_api("openai", "sk-...")

# Next time: auto-loads from secure storage
trainer.configure_api("openai")

# Generate training data
result = trainer.generate_api_training_data(
    tasks=["chat", "code", "avatar"],
    examples_per_task=100
)

# Full workflow: generate + train
config = trainer.train_from_api_data(
    tasks=["chat", "avatar"],
    examples_per_task=100,
    model_name="my_assistant"
)
```

### Secure Key Management
```python
from enigma_engine.utils.api_key_encryption import store_api_key, get_api_key

store_api_key("openai", "sk-...", "My key")
key = get_api_key("openai")
```

---

## Ideas for Later

### API & Training Enhancements
- [x] **API key rotation** - Auto-switch to backup key on rate limit/error
- [x] **Cost estimation** - Show estimated API cost before generating data
- [x] **Batch API calls** - Use batch APIs for lower cost (`core/batch_api.py`)
- [x] **Training data quality scoring** - Rate generated examples, filter low-quality
- [x] **Incremental training** - Add more data to existing trained model (`core/incremental_training.py`)
- [x] **Training checkpoints** - Save/resume training mid-way (`core/checkpointing.py`)
- [x] **Multi-GPU training support** - Distributed training for larger models (`core/multi_gpu.py`)
- [x] **Training data deduplication** - Remove duplicates before training
- [x] **Synthetic data augmentation** - Paraphrase, vary examples automatically (`core/augmentation.py`)
- [x] **Curriculum learning** - Start with easy examples, progress to harder (`core/curriculum_learning.py`)
- [x] **Mixed precision training** - BF16/FP16 for faster training with less VRAM (`core/mixed_precision.py`)
- [x] **Gradient checkpointing** - Trade compute for memory on large models (`core/gradient_checkpointing.py`)

### GUI Improvements
- [x] **API key import/export** - Backup and restore encrypted keys (`api_key_manager.py`)
- [x] **Training history** - Track all training runs with metrics (`training_history.py`)
- [x] **Live training visualization** - Loss curves, attention heatmaps (`gui/training_viz.py`)
- [x] **Model comparison tool** - Side-by-side model testing (`benchmarks.py`)
- [x] **Dark/light theme toggle** - User preference for GUI theme (`gui/themes.py`)
- [x] **Keyboard shortcuts** - Power user navigation (`gui/shortcuts.py`)
- [x] **Drag-and-drop training data** - Drop files directly into Build AI tab (`gui/drop_zone.py`)
- [x] **Quick model switcher** - Dropdown to swap active model without restart (`gui/model_switcher.py`)
- [x] **Chat export** - Export conversations as markdown/PDF/JSON (`utils/chat_export.py`)
- [x] **Natural language model config** - "Make it more creative" adjusts params (`core/nl_config.py`)
- [x] **Undo/redo in training** - Revert model to previous checkpoint (`core/checkpointing.py`)
- [x] **Split view chat** - Compare two models side-by-side in real-time (`gui/split_chat.py`)

### Model Features
- [x] **Model merging** - Combine models via SLERP, TIES, DARE, averaging (`core/model_merger.py`)
- [x] **LoRA/QLoRA loading** - Model has `load_lora()`/`merge_lora()` (training integration pending)
- [x] **Model quantization** - 4-bit/8-bit for smaller footprint (`core/quantization.py`)
- [x] **GGUF export** - Export TO GGUF format (`core/gguf_export.py`)
- [x] **HuggingFace Hub upload** - Already implemented (`push_to_hub()` in huggingface_exporter.py)
- [x] **Speculative decoding** - Use small model to draft, large to verify (`core/speculative_decoding.py`)
- [x] **Context extension** - RoPE scaling for longer context windows (`core/context_extension.py`)
- [x] **Sparse attention** - Efficient attention for very long contexts (`core/sparse_attention.py`)
- [x] **KV cache compression** - Reduce memory for long conversations (`core/kv_cache_compression.py`)
- [x] **Dynamic batching** - Batch multiple requests for throughput (`core/dynamic_batching.py`)
- [x] **Model benchmarking** - Automated performance testing (`core/benchmarks.py`)

### Avatar & Animation
- [x] **Real-time motion capture** - Webcam to avatar tracking with MediaPipe (`avatar/motion_capture.py`)
- [x] **Emotion detection from voice** - Infer emotion from audio/speech (`voice/emotion_detector.py`)
- [x] **Procedural idle animations** - Breathing, blinking, micro-movements (`avatar/ai_controls.py`)
- [x] **Avatar marketplace** - Share/download custom avatars (`avatar/marketplace.py`)
- [x] **Live2D support** - Import Live2D models for 2D VTuber avatars (`avatar/live2d.py`)
- [x] **Full body tracking** - VR controller support for body (`avatar/body_tracking.py`)
- [x] **Facial landmark to bones** - Map face mesh to avatar via motion_capture.py
- [x] **Unity/Unreal plugin** - Export avatar system to game engines (`integrations/unity_export.py`)
- [x] **Avatar clothing system** - Change outfits dynamically (`avatar/clothing.py`)
- [x] **Physics-based hair/cloth** - Realistic secondary motion (`avatar/physics_simulation.py`)
- [x] **Avatar performance modes** - Lightweight mode for low-end devices (`avatar/performance.py`)
- [x] **Detailed AI avatar controls** - Emotions, gestures, attention system (`avatar/ai_controls.py`)

### Voice & Audio
- [x] **Real-time voice conversion** - Change your voice to avatar voice (`voice/voice_conversion.py`)
- [x] **Whisper integration** - Already supported (install: `pip install openai-whisper`)
- [x] **Multi-speaker TTS** - Different voices in same output (`voice/multi_speaker.py`)
- [x] **Emotion-controlled TTS** - "Say this happily/sadly" (`voice/emotion_tts.py`)
- [x] **Background music mixing** - Auto-duck when speaking (`voice/audio_mixer.py`)
- [x] **Audio-reactive avatar** - Avatar moves to music/voice energy (`avatar/audio_reactive.py`)
- [x] **Voice profile sharing** - Export/import cloned voices (`voice/voice_profile.py`)
- [x] **Singing synthesis** - TTS for songs with melody (`voice/singing.py`)
- [x] **Real-time translation** - Speak English, output Japanese TTS (`i18n/translations.py`)
- [x] **Audio bookmarks** - Mark important moments in voice chat (`voice/audio_bookmarks.py`)

### Memory & RAG
- [x] **Semantic memory search** - Ask "What did I say about X last week?" (`memory/semantic_search.py`)
- [x] **Knowledge graph visualization** - See connections between memories (`memory/knowledge_graph.py`)
- [x] **Automatic summarization** - Compress old conversations (`memory/summarization.py`)
- [x] **Memory importance scoring** - Prioritize what to remember (`memory/summarization.py`)
- [x] **Cross-session context** - AI remembers across restarts (`memory/cross_session.py`)
- [x] **Document ingestion pipeline** - PDF/DOCX/web to memory (`tools/document_ingestion.py`)
- [x] **Memory backup/restore** - Export all memories encrypted (`memory/backup.py`)
- [x] **Forgetting mechanism** - Deliberately remove memories (`memory/forgetting.py`)
- [x] **Memory attribution** - Show which memory influenced response (`memory/rag.py`)
- [x] **External knowledge bases** - Connect to Wikipedia, Wikidata (`tools/game_wiki.py`)

### Multi-Agent System
- [x] **Agent templates** - Pre-built agents (researcher, coder, critic) (`agents/templates.py`)
- [x] **Agent communication protocols** - Structured message passing (`agents/protocols.py`)
- [x] **Task decomposition** - Break complex tasks into agent subtasks (`agents/task_decomposition.py`)
- [x] **Agent voting** - Multiple agents vote on best answer (`agents/voting.py`)
- [x] **Agent specialization training** - Fine-tune agents for their roles (`agents/templates.py`)
- [x] **Visual agent workspace** - See agents thinking/collaborating (`agents/visual_workspace.py`)
- [x] **Agent persistence** - Save/load agent states (`agents/persistence.py`)
- [x] **Inter-agent tool sharing** - One agent's output feeds another (`agents/tool_sharing.py`)
- [x] **Agent debate mode** - Agents argue different positions (`agents/debate.py`)
- [x] **Agent tournament** - Compete agents on benchmarks (`agents/tournament.py`)

### Gaming & Overlay
- [x] **Game profile auto-detection** - Recognize game from window/process (`tools/game_detector.py`)
- [x] **Quick commands** - "!build house" triggers game-specific action (`tools/game_commands.py`)
- [x] **Stream integration** - Show AI responses in OBS overlay (`integrations/obs_streaming.py`)
- [x] **Game state awareness** - Read screen/memory for context (`tools/game_state.py`)
- [x] **Achievement tracking** - AI congratulates on milestones (`tools/achievement_tracker.py`)
- [x] **Session stats** - Track gaming time, deaths, progress (`tools/game_stats.py`)
- [x] **Voice command gaming** - "Go left", "Use item" voice control (`tools/voice_gaming.py`)
- [x] **Multiplayer awareness** - Know when in party/raid (`tools/multiplayer_awareness.py`)
- [x] **Game-specific knowledge** - Auto-load wikis per game (`tools/game_wiki.py`)
- [x] **Replay analysis** - AI reviews recorded gameplay (`tools/replay_analysis.py`)

### Robotics & IoT
- [x] **ROS2 native nodes** - Full ROS2 Python package (`tools/ros2_integration.py`, `robotics/ros_integration.py`)
- [x] **SLAM visualization** - Show robot's map building (`robotics/slam.py`)
- [x] **Manipulation planning** - Pick and place with arm robots (`robotics/manipulation.py`)
- [x] **Multi-robot coordination** - Control robot swarms (`tools/robot_modes.py`)
- [x] **Sensor fusion display** - Visualize camera + lidar + IMU (`robotics/slam.py`)
- [x] **Home Assistant integration** - Control smart home devices (`integrations/home_assistant.py`)
- [x] **Voice-controlled robotics** - "Robot, go to the kitchen" (`tools/voice_gaming.py`)
- [x] **Simulation mode** - Test in Gazebo before real robot (`robotics/ros_integration.py`)
- [x] **Safety zones** - Define no-go areas for robot (`tools/robot_modes.py`)
- [x] **Battery management** - Auto-dock on low power (`tools/robot_modes.py`)

### Platform & Distribution
- [x] **Training data marketplace** - Share/download community packs (`marketplace/marketplace.py`)
- [x] **Model inheritance** - Fork existing AIs, keep attribution (`core/model_fork.py`)
- [x] **Web interface for training** - Browser-based training dashboard (`web/training_dashboard.py`)
- [x] **Mobile app integration** - iOS/Android companion app (`comms/mobile_api.py`)
- [x] **Docker production image** - Optimized container for deployment (`Dockerfile`, `docker-compose.yml`)
- [x] **Cloud training** - One-click training on cloud GPUs (`core/cloud_training.py`)
- [x] **Electron app packaging** - Single .exe for Windows/Mac/Linux (`packaging/electron/`)
- [x] **Auto-updater** - Check for updates on startup (`utils/auto_update.py`)
- [x] **Crash reporting** - Anonymous error collection for debugging (`utils/crash_reporter.py`)
- [x] **Usage telemetry** - Opt-in metrics for improvement (`web/telemetry_dashboard.py`)
- [x] **Plugin system** - Community can extend functionality (`core/plugin_system.py`)
- [x] **CLI chat mode** - Pure terminal interface for SSH/headless (`cli/chat.py`)

### Security & Privacy
- [x] **Content rating system** - Configurable SFW/NSFW modes with toggle in settings
- [x] **NSFW content filter** - Detect/block explicit content in images and text  
- [x] **Local-only mode** - Disable all network features (`utils/local_only.py`)
- [x] **Conversation encryption** - E2E encrypt stored chats (`memory/encryption.py`)
- [x] **PII scrubbing** - Auto-redact personal info in training data (`utils/pii_scrubbing.py`)
- [x] **Audit logging** - Track all API calls with timestamps (`utils/audit_logger.py`)
- [x] **Role-based access** - Admin/user permissions for multi-user (`utils/rbac.py`)
- [x] **API key scoping** - Limit which features each key can use (`utils/rbac.py`)
- [x] **Sandboxed code execution** - Run AI-generated code safely (`modules/sandbox.py`)
- [x] **Data export for GDPR** - Export all user data on request (`security/gdpr.py`)

### Advanced AI Features
- [x] **Reinforcement learning from feedback (RLHF)** - Human preference training (`core/rlhf.py`)
- [x] **Constitutional AI** - Self-critique and revision (`core/constitutional_ai.py`)
- [x] **Multi-modal training** - Images + text together (`core/multimodal.py`)
- [x] **Streaming inference API** - Token-by-token response streaming (`comms/streaming_api.py`)
- [x] **Tool calling training** - Function calling fine-tuning (`core/tool_calling_training.py`)
- [x] **RAG integration** - Already implemented in `enigma_engine/memory/rag.py`
- [x] **Chain-of-thought prompting** - Guide reasoning step-by-step (`core/chain_of_thought.py`)
- [x] **Self-play training** - AI debates itself to improve (`core/self_play.py`)
- [x] **Distillation** - Compress large model knowledge into small model (`core/distillation.py`)
- [x] **Ensemble inference** - Multiple models vote on output (`core/ensemble.py`)
- [x] **Uncertainty estimation** - AI knows when it's unsure (`core/uncertainty.py`)
- [x] **Active learning** - AI asks for labels on uncertain examples (`core/active_learning.py`)

### Accessibility
- [x] **Screen reader support** - Full accessibility for blind users (`gui/screen_reader.py`)
- [x] **High contrast themes** - For low vision users (`gui/accessibility_themes.py`)
- [x] **Font size scaling** - UI-wide zoom (`gui/accessibility.py`)
- [x] **Voice-only mode** - Operate entirely by voice (`voice/voice_only_mode.py`)
- [x] **Simplified UI mode** - Hide advanced features for beginners (`gui/simplified_mode.py`)
- [x] **Internationalization** - Translate UI to other languages (`gui/i18n.py`)
- [x] **RTL language support** - Arabic, Hebrew layout (`gui/rtl_support.py`)
