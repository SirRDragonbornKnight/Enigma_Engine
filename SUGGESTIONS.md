# Suggestions

**Date:** March 26, 2026 (Pass 10 — full re-review)

---

## Architecture — Verified Strengths (no action needed)

- **Model architecture** — LLaMA-class (RMSNorm, SwiGLU, RoPE, GQA, MoE). On-demand causal mask, vocab padding to 64, weight tying, KV cache with O(1) writes.
- **Training pipeline** — EMA, sequence packing, reasoning-weighted loss, PPO, validation loop, rolling checkpoints, early stopping.
- **Generation** — All native paths through `_generate_manual()` → `_sample_token()`. Sign-aware rep penalty, penalty-before-temperature, stop-string holdback.
- **Tokenizer protocol** — `TokenizerProtocol` with 4 implementations + HuggingFace + tiktoken. Named special token IDs.
- **API server** — CORS opt-in only, optional API key, no middleware.

---

## Known Limitations (won't fix)

| Item | Why |
|------|-----|
| `widgets.py`, `gui_mod_page.py`, `gui_docs_page.py` access private CTk internals (`._textbox`) | CustomTkinter has no public API for undo/textbox config |
| `desktop.py` `_is_process_alive` and `gui_logic_media.py` `_open_file` are Windows-only | Cross-platform needed only if Linux/macOS is a target |
| Tokenizer special token IDs differ across implementations (`<think>`=4/10/13) | Works if callers use instance attributes, not hardcoded IDs |
| `inference.py` modifies global `os.environ['PATH']` in `_load_gguf()` | Needed for llama.cpp CUDA DLL discovery |
| Lock acquisition pattern inconsistency in `engine_chat.py` | GGUF path: caller holds lock. Vision/native: callee holds lock. Both work |
| `builtin_commands.py` `file_append` and `train_data_add` use `open("a")` | Append-mode writes don't corrupt existing data on crash |
| `gui_pages_config.py` backup restore uses `write_bytes()` | One-time manual restore operation, not ongoing persistence |
| `gguf_dequant.py` skips Q4_1/Q5_0/Q5_1 tensors with warning | Manual dequant fallback path; primary GGUF path uses llama-server subprocess. Uncommon quantization types |

---

## Deferred (revisit at scale)

| Item | Trigger | Current State |
|------|---------|---------------|
| RAG on memory | Fact count exceeds 500 | **Blocked** — `MAX_FACTS=200` hard cap. Raise cap first, then assess need |
| Model-generated history summaries | Student model reaches 600M+ | Not yet — largest local model is ~165M (`base` preset) |
| Per-user emotional state via profiles | Multi-user deployment | Not yet — single-user desktop app |
| Clipboard command macOS support (`pbcopy`) | macOS user request | Not yet — Windows-primary, partial Linux. No macOS users |
| BPE UTF-8 mode flag validation on load | Non-ASCII tokenizer bug reported | **Latent risk** — `use_utf8_bytes=False` default silently drops non-latin-1 chars to `<unk>`. No bug report yet but any non-English training hits this |

---

## Suggestions (actionable)

### Fix (bugs / resource leaks)

| # | File | Issue | Priority |
|---|------|-------|----------|
| ~~70~~ | `rl_training.py` | ~~GPU memory leak: ref model not in try/finally~~ | **FIXED** |
| ~~71~~ | `vision_encoder.py` | ~~VideoCapture resource leak in encode_video_frames()~~ | **FIXED** |
| ~~72~~ | `training_queue.py` | ~~Queue save failure logged at DEBUG~~ | **FIXED** |

### Improve (performance / robustness)

| # | File | Issue | Priority |
|---|------|-------|----------|
| ~~73~~ | `model.py` | ~~Causal mask created on CPU~~ | **FIXED** |
| ~~74~~ | `gguf.py` | ~~Unbounded array allocation in GGUF parser~~ | **FIXED** |
| ~~75~~ | `gguf.py` | ~~Unbounded string allocation in GGUF parser~~ | **FIXED** |

### Delete (dead code)

| # | File | Issue | Priority |
|---|------|-------|----------|
| ~~76~~ | `builtin_commands.py` | ~~Unused Dict import~~ — **FALSE POSITIVE**: `Dict` is used in annotation strings throughout the file (20+ function signatures) | N/A |

### Research Implementations (from RESEARCH_REFERENCES.md — papers worth implementing NOW)

**Generation Quality — Missing Sampling Methods**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 77 | `engine_generation.py` | [Typical Sampling](https://arxiv.org/abs/2202.00666) (Meister et al.) | Add typical sampling: keep tokens near expected information content (conditional entropy). Reduces both boring repetition and incoherent randomness. ~30 lines in `_sample_token()` | Better output quality — competitive with top-p but reduces degenerate repetitions |
| 78 | `engine_generation.py` | [Mirostat](https://arxiv.org/abs/2007.14966) (Basu et al.) | Add Mirostat decoding: feedback-based adaptive top-k that targets a specific perplexity. Self-adjusting — no temperature tuning needed. ~50 lines | Consistent output quality regardless of prompt. Avoids boredom trap (low k) and confusion trap (high k) |
| 79 | `engine_generation.py` | Frequency + Presence Penalty (OpenAI-style) | Replace single repetition penalty with separate frequency penalty (how many times repeated) + presence penalty (seen at all). More nuanced control | Better repetition control — penalize "the the the" without penalizing legitimate name repetition |

**Sentiment — Current Gaps**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 80 | `sentiment.py` | Contextual Negation | Add 3-word negation window: "not happy" should flip polarity. Currently "happy" scores positive even after "not". ~20 lines | Fixes a real accuracy bug — negated sentiments score backwards |
| 81 | `sentiment.py` | VADER-style Upgrade | Expand word lists from ~80 words to 2000+ with valence ratings. Add intensifiers ("very", "extremely") and degree adverbs. Existing heuristic architecture stays, just better data | Major quality improvement from ~80 to 2000+ rated words |

**Training — Free Quality Improvements**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 82 | `training.py` | [R-Drop](https://arxiv.org/abs/2106.14448) (Liang et al.) | Add optional R-Drop: run each sample through model twice (different dropout masks), minimize KL divergence between the two outputs. ~20 lines in training loop. Config flag `r_drop_alpha` | Free quality boost — same data, same compute, better generalization. NeurIPS 2021 |
| 83 | `training.py` | [Gradient Noise](https://arxiv.org/abs/1511.06807) (Neelakantan et al.) | Add optional decaying Gaussian noise to gradients: `grad += N(0, η/(1+t)^γ)`. ~10 lines. Config flag `gradient_noise_eta` | Helps escape sharp minima. Especially useful for small models. Easy regularization |
| 84 | `training.py` | [SWA](https://arxiv.org/abs/1803.05407) (Izmailov et al.) | Extend existing EMA with SWA option: average weights from multiple points along training trajectory (not just exponential). ~30 lines extending EMAWeightAverager | Better generalization than EMA alone. Finds wider optima |
| 85 | `training.py` | [Cosine w/ Warm Restarts](https://arxiv.org/abs/1608.03983) (Loshchilov et al.) | Add SGDR scheduler option: periodic LR restarts that re-explore the loss landscape. `CosineAnnealingWarmRestarts` is already in PyTorch | Escapes local minima. Each restart finds different solution — combine with SWA |
| 86 | `bpe_tokenizer.py` | [BPE-Dropout](https://arxiv.org/abs/1910.13267) (Provilkov et al.) | During training, randomly skip BPE merges with probability p (e.g. 0.1). Use standard BPE at inference. ~15 lines in `encode()` | More robust to typos and rare words. Subword regularization for free |

**RL Training — Simpler Alternatives**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 87 | `rl_training.py` | [SimPO](https://arxiv.org/abs/2405.14734) (Meng et al.) | Add SimPO trainer: reference-model-free DPO using average log probability as implicit reward. ~100 lines as new trainer class | Halves VRAM during DPO training. Outperforms DPO by 6+ points on benchmarks. NeurIPS 2024 |
| 88 | `rl_training.py` | [GRPO](https://arxiv.org/abs/2501.12948) (DeepSeek) | Add GRPO trainer: no value head, groups of responses ranked relatively. ~120 lines | Eliminates value head entirely. Simpler than PPO, saves VRAM. Published in Nature |
| 89 | `rl_training.py` | [KTO](https://arxiv.org/abs/2402.01306) (Ethayarajh et al.) | Add KTO trainer: works with unpaired thumbs-up/down feedback, no matched pairs needed. ~80 lines | Easier data collection — don't need chosen/rejected pairs. ICML 2024 |
| 90 | `rl_training.py` | Adaptive KL | Change `kl_coeff` from fixed 0.1 to dynamic: increase when KL exceeds target, decrease when below. ~15 lines in PPO update loop | Auto-tunes KL penalty. Better RL stability |

**Inference Speed**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 91 | `model.py` / speculative | Adaptive Speculation K | Dynamically adjust `num_speculative_tokens` based on rolling acceptance rate. High acceptance → try more tokens. Low → fewer. ~20 lines | Self-tuning speculative decoding. Eliminates manual K tuning |
| 92 | `engine_chat.py` | Prompt/Prefix Caching | Cache KV states for system prompt between chat turns. System prompt rarely changes. Check if system prompt matches previous, reuse cached KVs | Instant speedup for chat — skip re-encoding the system prompt every turn |
| 93 | `inference.py` | torch.compile for inference | Currently only used in training. Enable `torch.compile(model, mode='reduce-overhead')` for inference too. Already have the infrastructure | 10-20% inference speedup on CUDA. Already proven in training path |

**Continual Learning — Anti-Forgetting**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 94 | `training.py` | [EWC](https://arxiv.org/abs/1612.00796) (Kirkpatrick et al.) | Add optional EWC regularization: compute Fisher Information matrix after initial training, penalize changes to important weights during subsequent fine-tuning. ~60 lines + config | Prevents catastrophic forgetting when adding new training data. Critical for ongoing desktop training |

**Progressive Growing — Missing Feature**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 95 | `progressive_growing.py` | [Gradual Unfreezing](https://arxiv.org/abs/1801.06146) (Howard & Ruder) | After growth, freeze old layers and gradually unfreeze top→bottom over N epochs. ~30 lines | Prevents catastrophic forgetting after model growth. Currently missing |

**RAG Improvements**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 96 | `rag.py` | BM25+ | Change IDF formula: add lower-bounding to prevent long-doc unfair penalization. One-line formula change | Better retrieval for longer documents. Drop-in fix |
| 97 | `rag.py` | Adaptive Chunking | Replace fixed 512-char chunks with sentence/paragraph-boundary snapping. Detect sentence boundaries before splitting | Chunks that respect document structure retrieve better |

**Structured Output**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 98 | `engine_generation.py` | [FSM/Grammar-Guided](https://arxiv.org/abs/2307.09702) (Outlines) | Add optional grammar/regex-constrained decoding: build token mask from FSM states, apply during sampling. ~150 lines | Guarantees valid JSON/command output. Eliminates malformed tool calls |

**Fine-Tuning Upgrades**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 99 | `lora_utils.py` | [DoRA](https://arxiv.org/abs/2402.09353) (Liu et al.) | Add DoRA option: decompose weights into magnitude + direction, apply LoRA to direction only. Same inference cost as LoRA. ~40 lines | Closes accuracy gap between LoRA and full fine-tuning. ICML 2024 Oral |

**Model Compression — Post-Training**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 100 | new `pruning.py` | [Wanda](https://arxiv.org/abs/2306.11695) (Sun et al.) | Add one-shot pruning: prune weights by magnitude × input activation, per output. No retraining needed. ~80 lines | Compress trained models 2-4x for faster desktop inference. ICLR 2024 |

**Test-Time Compute — Think Harder**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 101 | `engine_generation.py` | [s1 Budget Forcing](https://arxiv.org/abs/2501.19393) (Muennighoff et al.) | Implement budget forcing for `<think>` tokens: append "Wait" to force model to double-check answers before responding. Control think budget via min/max think tokens. ~40 lines | s1-32B beat o1-preview on math. Forces self-correction. Directly uses our existing `<think>` token infrastructure |
| 102 | `engine_generation.py` | [Test-Time Compute Scaling](https://arxiv.org/abs/2408.03314) (Snell et al.) | Add compute-optimal test-time strategy: adaptively allocate thinking budget per prompt difficulty. Easy prompts get short `<think>`, hard prompts get long. ~60 lines | 4x more efficient than best-of-N. Small model + more thinking can beat 14x larger model |

**Vision Encoder — Better Backbones**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 103 | `vision_encoder.py` | [SigLIP](https://arxiv.org/abs/2303.15343) (Zhai et al.) | Add SigLIP as pretrained vision backbone option alongside timm models. Sigmoid loss = better than CLIP, scales to smaller batch sizes | Better vision features than CLIP. ICCV 2023 Oral |
| 104 | `vision_encoder.py` | [TinyViT](https://arxiv.org/abs/2207.10666) (Wu et al.) | Add TinyViT-21M as lightweight vision encoder option. 84.8% ImageNet at 21M params | Perfect for desktop — small, fast, accurate. ECCV 2022 |

**Quantization — Post-Training Compression**

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 105 | `gguf.py` / new `quantize.py` | [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al.) | Add GPTQ quantization as export option alongside GGUF. One-shot 4-bit with calibration data. ~150 lines | Better quality than naive rounding at same compression. Industry standard alongside GGUF. ICLR 2023 |

**3D Avatar — Port & Upgrade Legacy Features**

The legacy branch (`enigma_engine/avatar/`) had ~3,500 lines across 20+ files: `lip_sync.py`, `emotion_sync.py`, `adaptive_animator.py`, `bone_control.py`, `ai_bridge.py`, gesture vocabulary (wave, nod, shake_head, jump, think), 9 expression sprites, 3 renderers, desktop pet, VRM/Blender support. These items port and upgrade those features into the current mod architecture (`mods/avatar/`), keeping full decoupling (subprocess + TCP, removable without affecting engine).

| # | Target File | Paper | What To Do | Impact |
|---|------------|-------|------------|--------|
| 106 | `mods/avatar/` | [SadTalker](https://arxiv.org/abs/2211.12194) + [EMOTE](https://arxiv.org/abs/2306.08990) | Port legacy `lip_sync.py` phoneme→mouth-sprite mapping into the mod's bone system. Map phonemes to jaw/lip bone rotations (15 viseme states). Legacy had sprite-based mouth animation — upgrade to bone-driven using the existing 50+ bone skeleton. Papers improve accuracy beyond the original rule-based approach | Restores lip sync that legacy had. Bone-driven is smoother than sprite-swapping |
| 107 | `mods/avatar/` | [EMOTE](https://arxiv.org/abs/2306.08990) | Port legacy `emotion_sync.py` into the mod. Legacy monitored AI text output for emotion keywords and auto-triggered expression changes. Current mod has `avatar.expression` but no auto-triggering. Wire engine sentiment → TCP event → expression blending. All avatar logic stays in the mod — engine just emits events | Restores auto-expression sync that legacy had. EMOTE's disentanglement improves on the keyword-matching approach |
| 108 | `mods/avatar/` | [MoGlow](https://arxiv.org/abs/1905.06598) + [MotionDiffuse](https://arxiv.org/abs/2208.15001) | Port legacy `adaptive_animator.py` idle animations into the mod. Legacy detected model capabilities (has arms? can blink?) and played appropriate fallback animations. Start with the same procedural approach (breathing, weight shifts, head tilts), upgrade to learned model later | Restores idle animation that legacy had. Avatar looks alive instead of frozen |
| 109 | `mods/avatar/` | [MDM](https://arxiv.org/abs/2209.14916) + [InsActor](https://arxiv.org/abs/2312.17135) | Port legacy `ai_bridge.py` + gesture vocabulary into the mod. Legacy had direct Python API (`avatar.control("wave")`). Rebuild as TCP command vocabulary — model emits `[avatar:wave]`, engine parses and sends via router. ~20 named gestures mapped to bone animation sequences | Restores AI-driven gestures that legacy had. TCP keeps mod decoupled unlike legacy's direct import |

**Legacy Subsystem Conversions (port legacy branch to mod architecture)**

Each subsystem from the [legacy branch](https://github.com/SirRDragonbornKnight/Enigma_AI_Engine/tree/legacy/enigma_engine) becomes a proper mod in `mods/`. Same pattern as existing mods: `mod.json` + `main.py` + subprocess + TCP (port 99XX) + removable. Each gets its own GUI page. Avatar (#106-#109 above) already has paper-backed upgrade items — these items cover the remaining subsystems.

| # | Legacy Source | Target Mod | What To Port | Scale |
|---|-------------|------------|-------------|-------|
| 110 | `enigma_engine/voice/` | `mods/voice/` | Port remaining legacy voice features into existing voice mod. Legacy had 11 files: `audio_analyzer.py`, `listener.py`, `natural_tts.py`, `stt_simple.py`, `tts_simple.py`, `vad.py`, `voice_generator.py`, `voice_pipeline.py`, `voice_profile.py`, `whisper_stt.py`, `__init__.py`. Current mod (port 9907) has basic TTS/STT. Missing: audio analysis, VAD, voice profiles, voice pipeline orchestration, natural TTS, whisper integration. Add TCP commands for each. GUI page for voice settings, engine selection, voice profiles | 11 files → upgrade existing mod |
| 111 | `enigma_engine/tools/` | `mods/tools/` or split | Port 35+ tool files. Legacy had: `browser_tools.py`, `gaming_tools.py`, `iot_tools.py`, `robot_tools.py`, `pi_robot.py`, `robot_modes.py`, `motion_tracking.py`, `simple_ocr.py`, `analytics.py`, `bias_detection.py`, `automation_tools.py`, `productivity_tools.py`, `media_tools.py`, `knowledge_tools.py`, `data_tools.py`, `document_tools.py`, `communication_tools.py`, `interactive_tools.py`, `vision.py`, `web_tools.py`, `file_tools.py`, `self_tools.py`, `system_tools.py`, `memory_tools.py`, `game_router.py`, `streaming.py`, `code_style_analyzer.py`, `url_safety.py`, `async_executor.py`, `cache.py`, `parallel.py`, `rate_limiter.py`, `permissions.py`, `validation.py`, `tool_definitions.py`, `tool_executor.py`, `tool_manager.py`, `tool_registry.py`, `plugins.py`, `avatar_tools.py`. Consider splitting by domain into separate mods (e.g. `mods/browser_tools/`, `mods/gaming/`, `mods/robot/`) or keep as one. Create mod.json with TCP commands per tool category. GUI page with tool categories and enable/disable per tool | 35+ files — evaluate split vs. single mod |
| 112 | `enigma_engine/web/` | `mods/web/` | Port web interface: `server.py` (FastAPI), `app.py` (Flask legacy), `auth.py` (token-based), `discovery.py` (mDNS/Bonjour), `static/` (HTML, JS, CSS, service worker, PWA manifest, icons), `templates/`. Features: WebSocket real-time chat, QR code connection, mobile responsive PWA, local network discovery, REST API endpoints. Create mod.json with TCP commands for server start/stop/status, port config, auth management. GUI page for web server settings (enable, port, auth toggle, QR code display) | 8 files + static assets |
| 113 | `enigma_engine/comms/` | `mods/comms/` | Port communications layer: `api_server.py`, `remote_client.py`. API server for remote engine access, remote client for connecting to other Enigma instances. Evaluate overlap with web mod (#112) — may merge or keep separate (comms = engine-to-engine, web = browser UI). Create mod.json with TCP commands for remote connections. GUI page for connection management | 3 files — evaluate merge with web mod |
| 114 | `enigma_engine/learning/` | `mods/learning/` | Port federated learning system. Legacy README documents full architecture: `FederatedLearning` (modes: opt-in/opt-out/disabled), `WeightUpdate` (signed weight deltas), `DifferentialPrivacy` (epsilon/delta configurable), `SecureAggregator` (simple/weighted/median), `TrustManager` (device reputation, byzantine detection, banning), `DataFilter` (PII removal, content filtering), `TrainingCoordinator` (round management, peer callbacks). Privacy levels: none/low/medium/high/maximum. Verify what's implemented vs. designed in legacy `__init__.py`, then build out as mod. Create mod.json with TCP commands for FL rounds, peer discovery, privacy settings. GUI page for federated config (mode, privacy level, aggregation method, trust threshold) | README + __init__.py (design complete, implementation TBD) |
| 115 | `enigma_engine/modules/` | `mods/modules/` or integrate | Port module management: `manager.py`, `registry.py`, `sandbox.py`, `updater.py`, `docs.py`. This is infrastructure for managing other modules/mods. Evaluate whether this becomes a standalone mod or merges into the engine's existing mod system (`router.py`, `gui_mods.py`, `gui_mod_page.py`). May overlap with what `mod_tools.py` and `plugin_loader.py` already do | 6 files — evaluate overlap with existing mod system |

**Legacy stubs (planned but not implemented — just `__init__.py`):**

| # | Legacy Source | Target Mod | Status |
|---|-------------|------------|--------|
| 116 | `enigma_engine/companion/` | `mods/companion/` | Stub only. No implementation in legacy. Design and build as new mod when companion features are needed |
| 117 | `enigma_engine/self_improvement/` | `mods/self_improvement/` | Stub only. Current engine already has self-improvement via adaptive training, curated dataset auto-accumulate, and learn-while-chatting. Evaluate what this mod would add beyond existing features |
| 118 | `enigma_engine/mobile/` | `mods/mobile/` | Stub only. May overlap with web mod (#112) for mobile access via PWA. Design as separate mod if native mobile features are needed beyond web |

**Pending: Paper research per mod** — Avatar papers already added (#106-#109). Voice, tools, and federated learning mods need dedicated paper research passes in RESEARCH_REFERENCES.md to find papers that improve their implementations (e.g., VITS/VALL-E for voice TTS, Toolformer/ReAct for tool use, FedAvg/FedProx for federated learning).

---

## AI Research References

Moved to [RESEARCH_REFERENCES.md](RESEARCH_REFERENCES.md) — 204 references across 29 categories, with priority tiers and paper links.
33 research items (#77-#109) + 9 legacy conversion items (#110-#118) added to Suggestions above.

---

## Code Review History

Passes 1-7 (March 25): 69 items found and fixed. All resolved.
Pass 8 (March 25): Full re-review of all ~65 files. 0 new bugs. 11 candidate findings all verified as false positives.
Pass 9 (March 25): Fresh review + cleanup. Fixed 2 font offset test failures (tests now reset state before asserting). Moved research references to dedicated file. 2396 passed, 0 failed, 3 skipped.
Pass 10 (March 26): Full re-review of all ~65 source files + 23 GUI files. 7 findings (#70-#76): 6 fixed, 1 false positive (#76). ~30+ subagent findings triaged as false positives. 2396 passed, 0 failed, 3 skipped.
Research Review (March 26): Cross-referenced 173 papers against actual codebase. Added priority tiers, paper links, 7 new papers. 24 actionable items (#77-#100) added to Suggestions.
Research Review 2 (March 26): Found 17 more missing papers — foundational refs (Transformer, LLaMA, RoPE, RMSNorm, SwiGLU, AdamW, RAG), quantization (GPTQ, AWQ, SmoothQuant), vision (SigLIP, DINOv2, TinyViT), test-time compute (s1, Snell), architecture (DeepSeek-V3, Qwen2.5). 5 new actionable items (#101-#105).
Research Review 3 (March 26): Added 3D Avatar & Animation section — 7 papers (FaceFormer, SadTalker, EMOTE, MoGlow, MotionDiffuse, MDM, InsActor). 4 actionable items (#106-#109) porting legacy avatar features (lip sync, emotion sync, idle animation, AI gestures) into the decoupled mod architecture with paper-backed upgrades.
Legacy Subsystem Review (March 26): Audited all legacy branch subsystems. 9 items (#110-#118): 6 real conversions (voice, tools, web, comms, learning, modules) + 3 stubs (companion, self_improvement, mobile). Paper research pending for voice, tools, and federated learning mods.
