# ForgeAI Suggestions Box

Ideas and improvements to work on. Check off when done!

---

## High Priority

- [ ] **LoRA Training Implementation** - Complete the TODO in `core/lora.py` and `core/inference.py` for adapter loading/training
- [ ] **Thread Safety in Camera Tab** - Add `QMutex` for `current_frame` access in `gui/tabs/camera_tab.py`
- [ ] **Worker Cleanup** - Add `closeEvent()` handlers to stop worker threads in tabs that spawn them

---

## Features

- [ ] **Streaming Response API** - Add SSE streaming to `/v1/chat/completions` endpoint
- [ ] **Model Download Progress** - Show progress bars when downloading HuggingFace models
- [ ] **Memory Usage Dashboard** - Live RAM/VRAM usage in GUI with per-module breakdown
- [ ] **Plugin System** - Allow custom modules without modifying core code
- [ ] **Dark/Light Theme Toggle** - Theme switching in settings
- [ ] **Export Chat History** - Export conversations to markdown/JSON
- [ ] **Batch Processing** - Queue multiple prompts for batch inference
- [ ] **Model Comparison** - Side-by-side comparison of different model responses

---

## Architecture

- [ ] **Singleton ModuleManager** - Use `get_manager()` consistently to prevent multiple registrations
- [ ] **Async/Await Migration** - Move from threading to `asyncio` for API server
- [ ] **Config Validation** - Add pydantic/dataclass validation for config files

---

## Ideas (Not Yet Planned)

Add new ideas here! Format: `- [ ] **Title** - Description`

- [ ] **Voice Cloning Improvements** - Better voice sample analysis
- [ ] **Multi-GPU Support** - Distribute model across multiple GPUs
- [ ] **Mobile App** - React Native companion app
- [ ] **Conversation Branching** - Fork conversations to explore different paths

---

## Completed

Move items here when done:

- [x] **ModuleManager Auto-Registration** - Modules now auto-register on init (2026-02-03)
- [x] **Scheduler Security** - Fixed command injection vulnerability (2026-02-03)
- [x] **ListDirectory Security** - Added path traversal protection (2026-02-03)

---

*Last updated: 2026-02-03*
